"""MQTT transport with optional mutual TLS for DeepUWF federation.

Payloads are JSON so the broker never receives executable Python objects.
"""

import json
import ssl
import threading
import time

import paho.mqtt.client as mqtt
import torch


class MQTTTransport:
    def __init__(self, args):
        self.args = args
        self.messages = []
        self.lock = threading.Lock()
        # paho-mqtt 2.x introduced an explicit callback API version; retain
        # compatibility with the 1.x release used by the original code.
        if hasattr(mqtt, "CallbackAPIVersion"):
            self.client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION1,
                                      client_id=args.client_id)
        else:
            self.client = mqtt.Client(client_id=args.client_id)
        self.client.on_message = self._on_message
        if args.mqtt_tls:
            if not (args.mqtt_ca_cert and args.mqtt_cert and args.mqtt_key):
                raise ValueError("--mqtt-tls requires --mqtt-ca-cert, --mqtt-cert, and --mqtt-key")
            self.client.tls_set(
                ca_certs=args.mqtt_ca_cert,
                certfile=args.mqtt_cert,
                keyfile=args.mqtt_key,
                tls_version=ssl.PROTOCOL_TLS_CLIENT,
            )
            self.client.tls_insecure_set(args.mqtt_tls_insecure)
        self.client.connect(args.mqtt_host, args.mqtt_port)
        self.client.subscribe(self.topic("updates", "+"), qos=args.mqtt_qos)
        self.client.loop_start()

    def topic(self, *parts):
        prefix = self.args.mqtt_topic.strip("/")
        return "/".join((prefix, self.args.run_id, *parts))

    def _on_message(self, _client, _userdata, message):
        try:
            payload = json.loads(message.payload.decode("utf-8"))
            if payload.get("protocol") != "deepuwf-federated-v1":
                return
            with self.lock:
                self.messages.append(payload)
        except (UnicodeDecodeError, json.JSONDecodeError, KeyError):
            return

    @staticmethod
    def _encode_state(state, keys):
        return {key: state[key].detach().cpu().tolist() for key in keys}

    @staticmethod
    def _decode_state(parameters):
        return {key: torch.tensor(value) for key, value in parameters.items()}

    def publish_update(self, round_index, state, keys, sample_count, metrics):
        payload = {
            "protocol": "deepuwf-federated-v1",
            "sender": self.args.client_id,
            "round": round_index,
            "sample_count": int(sample_count),
            "metrics": metrics,
            "parameters": self._encode_state(state, keys),
        }
        encoded = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        info = self.client.publish(self.topic("updates", self.args.client_id), encoded, qos=self.args.mqtt_qos)
        info.wait_for_publish()

    def collect_updates(self, round_index):
        deadline = time.monotonic() + self.args.mqtt_wait_seconds
        collected = {}
        while time.monotonic() < deadline:
            with self.lock:
                remaining = []
                for message in self.messages:
                    if message.get("round") == round_index and message.get("sender") != self.args.client_id:
                        collected[message["sender"]] = message
                    else:
                        remaining.append(message)
                self.messages = remaining
            if len(collected) >= self.args.min_peer_updates:
                break
            time.sleep(0.25)
        for update in collected.values():
            update["parameters"] = self._decode_state(update["parameters"])
        return list(collected.values())

    def close(self):
        self.client.loop_stop()
        self.client.disconnect()
