import argparse
import dataclasses


def dataclass_to_args(dataclass_obj):
    """converts a dataclass object into an argparse object"""
    parser = argparse.ArgumentParser()
    for field in dataclasses.fields(dataclass_obj):
        parser.add_argument(f"--{field.name}", default=getattr(dataclass_obj, field.name))
    args = parser.parse_args()
    return args
