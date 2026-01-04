#!/usr/bin/env python3
import os
import sys

BASE = os.path.dirname(os.path.dirname(__file__))  # /home/zhbs/sda/zyx/RL
FEATURE_FLAG = 'raptor_mini_preview'
GLOBAL_CONFIG = os.path.join(BASE, 'config', 'feature_flags.yml')
CLIENTS_DIR = os.path.join(BASE, 'clients')

def read_global_flag():
    try:
        with open(GLOBAL_CONFIG, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith(FEATURE_FLAG + ':'):
                    val = line.split(':', 1)[1].strip().lower()
                    return val in ('true', '1', 'yes', 'on')
    except FileNotFoundError:
        return False
    return False

def apply_to_client(client_path, enabled):
    cfg_path = os.path.join(client_path, 'config.yml')
    lines = []
    if os.path.exists(cfg_path):
        with open(cfg_path, 'r') as f:
            lines = f.readlines()
    found = False
    for i, l in enumerate(lines):
        if l.strip().startswith(FEATURE_FLAG + ':'):
            lines[i] = f"{FEATURE_FLAG}: {'true' if enabled else 'false'}\n"
            found = True
            break
    if not found:
        if lines and not lines[-1].endswith('\n'):
            lines[-1] = lines[-1] + '\n'
        lines.append(f"{FEATURE_FLAG}: {'true' if enabled else 'false'}\n")
    os.makedirs(os.path.dirname(cfg_path), exist_ok=True)
    with open(cfg_path, 'w') as f:
        f.writelines(lines)

def main():
    enabled = read_global_flag()
    if not os.path.isdir(CLIENTS_DIR):
        print(f"No clients directory at {CLIENTS_DIR}", file=sys.stderr)
        sys.exit(1)
    for name in sorted(os.listdir(CLIENTS_DIR)):
        path = os.path.join(CLIENTS_DIR, name)
        if os.path.isdir(path):
            apply_to_client(path, enabled)
            print(f"Set {FEATURE_FLAG}={enabled} for client {name}")
    print("Done.")

if __name__ == '__main__':
    main()
