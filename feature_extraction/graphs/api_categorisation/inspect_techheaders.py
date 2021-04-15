import yaml

# with open('allAPI_categories_techheaders.yaml', 'r') as f:
#     yaml_dict = yaml.load(f, Loader=yaml.FullLoader)

# for curr_key, curr_lst in yaml_dict.items():
#     for key, lst in yaml_dict.items():
#         if curr_key == key:
#             continue

#         if set(curr_lst) <= set(lst):
#             print('{} is a subset of {}'.format(curr_key, key))

#         if set(curr_lst) == set(lst):
#             print('!!!{} = {}!!!'.format(curr_key, key))

#===========

# print('Removing duplicates...')

# with open('allAPI_categories_techheaders.yaml', 'r') as f:
#     yaml_dict = yaml.load(f, Loader=yaml.FullLoader)

# remove = ['smartcrd', 'hcp_sdk', 'clushyperv_sdk', 'iscsitarg_sdk', 'whqlprov_sdk', 'wlbsprov_sdk', 'eventlogprov_sdk',
#           'oprec_sdk', 'input_pointerdevice_sdk', 'input_sourceid_sdk', 'input_touchhittest_sdk', 'input_touchinjection_sdk',
#           'inputmsg_sdk', 'msdtcwmi_sdk', 'winsensors_com_ref_sdk', 'winsensors_sdk', 'tapi2_sdk', 'vstor_sdk', 'sys/stat.h']

# for key in remove:
#     yaml_dict.pop(key)

# yaml_dict_reduced = dict(yaml_dict)
# for curr_key, curr_lst in yaml_dict.items():
#     for key, lst in yaml_dict.items():
#         if curr_key == key:
#             continue

#         if set(curr_lst) < set(lst):
#             for val in yaml_dict_reduced[key]:
#                 if val in curr_lst:
#                     yaml_dict_reduced[key].remove(val)

# for key, lst in yaml_dict_reduced.items():
#     if len(lst) == 0: 
#         print(key)

# with open('allAPI_categories_techheaders_reduced.yaml', 'w') as f:
#     yaml.dump(yaml_dict_reduced, f)

#============

with open('allAPI_categories_techheaders_reduced.yaml', 'r') as f:
    yaml_dict = yaml.load(f, Loader=yaml.FullLoader)

for curr_key, curr_lst in yaml_dict.items():
    for key, lst in yaml_dict.items():
        if curr_key == key:
            continue

        if set(curr_lst) <= set(lst):
            print('{} is a subset of {}'.format(curr_key, key))

        if set(curr_lst) == set(lst):
            print('!!!{} = {}!!!'.format(curr_key, key))