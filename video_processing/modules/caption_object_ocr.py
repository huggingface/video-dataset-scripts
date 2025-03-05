FLORENCE = None


def load_florence(
    hf_hub_or_path="microsoft/Florence-2-large",
    device="cpu",
    dtype="float32",
    check_task_types=True,
):
    global FLORENCE
    from florence_tool import FlorenceTool

    FLORENCE = FlorenceTool(
        hf_hub_or_path=hf_hub_or_path,
        device=device,
        dtype=dtype,
        check_task_types=check_task_types
    )
    FLORENCE.load_model()


def run(
    image,
    task_prompt,
):
    if FLORENCE is None:
        load_florence()
    return FLORENCE.run(
        image=image,
        task_prompt=task_prompt,
    )
