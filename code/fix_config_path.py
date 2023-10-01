# -*- coding: utf-8 -*-

# %%
from pathlib import Path
import yaml
import io

# %%
data_dir = Path.home() / 'TVT' / 'data' / 'dog_negishi'

rep_path = ['/home/hika/TVT_data', '/home/animallab/TVT/data']
for ff in data_dir.glob('*.mp4'):
    dirs = list(data_dir.glob(f"{ff.stem}-DLCGUI-*"))
    if len(dirs):
        for dd in dirs:
            conf_f = dd / 'config_rel.yaml'
            if not conf_f.is_file():
                continue

            # Read config file
            with open(conf_f, 'r') as stream:
                config_data = yaml.safe_load(stream)

            # Convert paths
            if '${DATA_ROOT}' not in config_data['project_path']:
                project_path = config_data['project_path']
                for rep in rep_path:
                    project_path = project_path.replace(rep, '${DATA_ROOT}')

                config_data['project_path'] = project_path

            video_sets = {}
            for vf0 in config_data['video_sets'].keys():
                if 'dog_negishi' in Path(vf0).parts:
                    ii = Path(vf0).parts.index('dog_negishi')
                    vf = '${DATA_ROOT}/' + '/'.join(Path(vf0).parts[ii:])
                    video_sets[vf] = config_data['video_sets'][vf0]
                elif '${DATA_ROOT}/' in vf0:
                    vf = vf0.replace('${DATA_ROOT}/',
                                     '${DATA_ROOT}/dog_negishi')
                else:
                    vf = vf0
                video_sets[vf] = config_data['video_sets'][vf0]
            config_data['video_sets'] = video_sets

            # Write config file
            with io.open(str(conf_f), 'w', encoding='utf8') as outfile:
                yaml.dump(config_data, outfile, default_flow_style=False,
                          allow_unicode=True)
