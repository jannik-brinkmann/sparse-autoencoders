import dataclasses
from dataclasses import dataclass, asdict, is_dataclass
import os

import json


class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        return super().default(o)
    

def save_config(config: dataclass, checkpoint_dir: str):
    filepath = os.path.join(checkpoint_dir, "config.json")
    with open(filepath, 'w') as f:
        json.dump(config, f, cls=EnhancedJSONEncoder, indent=4)
    