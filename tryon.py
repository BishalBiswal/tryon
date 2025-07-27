import os
import random
import sys
from typing import Sequence, Mapping, Any, Union
import torch


def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    """Returns the value at the given index of a sequence or mapping.

    If the object is a sequence (like list or string), returns the value at the given index.
    If the object is a mapping (like a dictionary), returns the value at the index-th key.

    Some return a dictionary, in these cases, we look for the "results" key

    Args:
        obj (Union[Sequence, Mapping]): The object to retrieve the value from.
        index (int): The index of the value to retrieve.

    Returns:
        Any: The value at the given index.

    Raises:
        IndexError: If the index is out of bounds for the object and the object is not a mapping.
    """
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]


def find_path(name: str, path: str = None) -> str:
    """
    Recursively looks at parent folders starting from the given path until it finds the given name.
    Returns the path as a Path object if found, or None otherwise.
    """
    # If no path is given, use the current working directory
    if path is None:
        path = os.getcwd()

    # Check if the current directory contains the name
    if name in os.listdir(path):
        path_name = os.path.join(path, name)
        print(f"{name} found: {path_name}")
        return path_name

    # Get the parent directory
    parent_directory = os.path.dirname(path)

    # If the parent directory is the same as the current directory, we've reached the root and stop the search
    if parent_directory == path:
        return None

    # Recursively call the function with the parent directory
    return find_path(name, parent_directory)


def add_comfyui_directory_to_sys_path() -> None:
    """
    Add 'ComfyUI' to the sys.path
    """
    comfyui_path = find_path("ComfyUI")
    if comfyui_path is not None and os.path.isdir(comfyui_path):
        sys.path.append(comfyui_path)
        print(f"'{comfyui_path}' added to sys.path")


def add_extra_model_paths() -> None:
    """
    Parse the optional extra_model_paths.yaml file and add the parsed paths to the sys.path.
    """
    try:
        from main import load_extra_path_config
    except ImportError:
        print(
            "Could not import load_extra_path_config from main.py. Looking in utils.extra_config instead."
        )
        from utils.extra_config import load_extra_path_config

    extra_model_paths = find_path("extra_model_paths.yaml")

    if extra_model_paths is not None:
        load_extra_path_config(extra_model_paths)
    else:
        print("Could not find the extra_model_paths config file.")


add_comfyui_directory_to_sys_path()
add_extra_model_paths()


def import_custom_nodes() -> None:
    """Find all custom nodes in the custom_nodes folder and add those node objects to NODE_CLASS_MAPPINGS

    This function sets up a new asyncio event loop, initializes the PromptServer,
    creates a PromptQueue, and initializes the custom nodes.
    """
    import asyncio
    import execution
    from nodes import init_extra_nodes
    import server

    # Creating a new event loop and setting it as the default loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Creating an instance of PromptServer with the loop
    server_instance = server.PromptServer(loop)
    execution.PromptQueue(server_instance)

    # Initializing custom nodes
    init_extra_nodes()


from nodes import NODE_CLASS_MAPPINGS


def main():
    import_custom_nodes()
    with torch.inference_mode():
        checkpointloadersimple = NODE_CLASS_MAPPINGS["CheckpointLoaderSimple"]()
        checkpointloadersimple_1 = checkpointloadersimple.load_checkpoint(
            ckpt_name="epicrealism_naturalSinRC1VAE.safetensors"
        )

        cliptextencode = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
        cliptextencode_2 = cliptextencode.encode(
            text="mens blue shirt", clip=get_value_at_index(checkpointloadersimple_1, 1)
        )

        cliptextencode_5 = cliptextencode.encode(
            text="disfigured, multiple fingers,blurred",
            clip=get_value_at_index(checkpointloadersimple_1, 1),
        )

        controlnetloader = NODE_CLASS_MAPPINGS["ControlNetLoader"]()
        controlnetloader_7 = controlnetloader.load_controlnet(
            control_net_name="outfitToOutfit_v20.safetensors"
        )

        controlnetloader_9 = controlnetloader.load_controlnet(
            control_net_name="control_sd15_openpose.pth"
        )

        loadimage = NODE_CLASS_MAPPINGS["LoadImage"]()
        loadimage_17 = loadimage.load_image(
            image="WhatsApp Image 2024-09-05 at 22.10.34_97428e31.jpg"
        )

        controlnetapplyadvanced = NODE_CLASS_MAPPINGS["ControlNetApplyAdvanced"]()
        ksampler = NODE_CLASS_MAPPINGS["KSampler"]()
        anything_everywhere = NODE_CLASS_MAPPINGS["Anything Everywhere"]()

        for q in range(1):
            controlnetapplyadvanced_8 = controlnetapplyadvanced.apply_controlnet(
                strength=1.0000000000000002,
                start_percent=0,
                end_percent=1,
                positive=["6", 0],
                negative=["6", 1],
                control_net=get_value_at_index(controlnetloader_9, 0),
                image=["10", 0],
            )

            ksampler_11 = ksampler.sample(
                seed=random.randint(1, 2**64),
                steps=35,
                cfg=6.5,
                sampler_name="dpmpp_2m",
                scheduler="karras",
                denoise=0.7500000000000001,
                model=get_value_at_index(checkpointloadersimple_1, 0),
                positive=get_value_at_index(controlnetapplyadvanced_8, 0),
                negative=get_value_at_index(controlnetapplyadvanced_8, 1),
                latent_image=["31", 0],
            )

            anything_everywhere_18 = anything_everywhere.func(
                anything=get_value_at_index(loadimage_17, 0)
            )

            anything_everywhere_19 = anything_everywhere.func(
                anything=get_value_at_index(checkpointloadersimple_1, 2)
            )


if __name__ == "__main__":
    main()
