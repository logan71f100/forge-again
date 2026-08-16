def register(options_templates, options_section, OptionInfo):
    options_templates.update(options_section((None, "Forge Hidden options"), {
        "forge_unet_storage_dtype": OptionInfo('Automatic'),
        "forge_inference_memory": OptionInfo(1024),
        "forge_async_loading": OptionInfo('Queue'),
        "forge_pin_shared_memory": OptionInfo('CPU'),
        "forge_preset": OptionInfo('sd'),
        "forge_additional_modules": OptionInfo([]),
    }))
    options_templates.update(options_section(('ui_alternatives', "UI alternatives", "ui"), {
        "forge_canvas_plain": OptionInfo(False, "ForgeCanvas: use plain background").needs_reload_ui(),
        "forge_canvas_toolbar_always": OptionInfo(False, "ForgeCanvas: toolbar always visible").needs_reload_ui(),
    }))
    options_templates.update(options_section(('system', "System", "ui"), {
        # OFF by default. On a normal install this instrumentation writes
        # client-debug.log into the webui root, logs every queue-stream event to
        # the browser console, and POSTs its ring buffer on the ping cadence --
        # all of which exist to diagnose "the generation never started" / "the
        # image never arrived", not to run in production. Turning it on costs
        # nothing until something breaks; leaving it on costs a growing log file
        # nobody reads.
        # This gates LOGGING ONLY: the connection watchdog, orphan recovery and
        # the submit-retry banner are functional and always run.
        "forge_client_forensics": OptionInfo(False, "Log connection forensics for debugging (writes client-debug.log)"),
    }))
