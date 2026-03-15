import threading


def timeout_input(prompt, default="", timeout=5, interactive=True):
    if not interactive:
        return default

    print("%s (%s秒内未输入将使用默认值: %s)" % (prompt, timeout, default))
    result = {"value": None}

    def _input_thread():
        try:
            result["value"] = input()
        except Exception:
            result["value"] = None

    t = threading.Thread(target=_input_thread, daemon=True)
    t.start()
    t.join(timeout)

    if t.is_alive():
        print("\n超时未输入，使用默认值: %s" % default)
        return default

    return result["value"] if result["value"] is not None else default
