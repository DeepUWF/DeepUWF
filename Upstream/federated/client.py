"""Federated training loop extracted and cleaned from the original DeepUWF client."""

import json
from pathlib import Path

from .aggregation import aggregate_partial_state, select_shareable_keys
from .mqtt_transport import MQTTTransport


def run_federated(args, model, model_without_ddp, optimizer, loss_scaler, criterion, data_dict, device, save_model):
    if args.distributed:
        raise ValueError("Use one DeepUWF federated client per site/GPU; do not combine --federated with torchrun.")
    if args.pretrain:
        from Model.pretrain import train_one_epoch
        evaluate = None
    else:
        from Model.finetune import train_one_epoch, evaluate

    transport = MQTTTransport(args)
    best_auc = float("-inf")
    try:
        for round_index in range(args.start_epoch, args.epochs):
            stats = train_one_epoch(model, optimizer, loss_scaler, data_dict, device, round_index, args, criterion=criterion)
            if evaluate is None:
                metrics = {"pretrain_loss": float(stats["loss"])}
                sample_count = len(data_dict["data_loader_train"].dataset)
            else:
                validation, auc = evaluate(model, data_dict, device, str(Path(args.log_dir) / args.task), "val",
                                           args.num_classes, multi_label=args.multi_label)
                metrics = {"accuracy": float(validation["acc1"]), "auc_roc": float(auc)}
                sample_count = len(data_dict["data_loader_train"].dataset)
                if auc > best_auc:
                    best_auc = auc
                    save_model(args=args, model=model, model_without_ddp=model_without_ddp,
                               optimizer=optimizer, loss_scaler=loss_scaler, epoch=round_index)

            state = model_without_ddp.state_dict()
            keys = select_shareable_keys(state, args.share_fraction, f"{args.seed}:{round_index}:{args.client_id}")
            transport.publish_update(round_index, state, keys, sample_count, metrics)
            updates = transport.collect_updates(round_index)
            if updates:
                local_update = {"sample_count": sample_count,
                                "parameters": {key: state[key].detach().cpu() for key in keys}}
                model_without_ddp.load_state_dict(aggregate_partial_state(state, [local_update, *updates]), strict=True)
            print(f"federated round {round_index}: shared {len(keys)} tensors; received {len(updates)} peer updates")
            with open(Path(args.output_dir) / "federated_log.jsonl", "a", encoding="utf-8") as handle:
                handle.write(json.dumps({"round": round_index, "local": metrics, "peer_updates": len(updates)}) + "\n")
    finally:
        transport.close()
    return best_auc
