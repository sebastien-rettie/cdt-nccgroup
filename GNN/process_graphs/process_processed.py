import argparse, os

import numpy as np


def main(input_dir):

    save_dir = os.path.join(input_dir, "small")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    total_files = len(os.listdir(input_dir))
    for idx, filename in enumerate(os.listdir(input_dir)):
        if filename.endswith('.npz'):    
            data = np.load(os.path.join(input_dir, filename))

            if(data["E"].shape[1] <= 5):
                os.rename(os.path.join(input_dir, filename), os.path.join(save_dir, filename))
                
            if ((idx + 1) % 500) == 0:
                print('[{}/{}] - ({:2.2%})'.format(idx + 1, total_files, float(idx + 1)/float(total_files)))



def parse_arguments():

    parser = argparse.ArgumentParser(description="Move graphs with < 5 edges to a subfolder.")
    parser.add_argument("input_dir")
    args = parser.parse_args()
    
    return (args.input_dir,)


if __name__ == "__main__":
    arguments = parse_arguments()
    main(*arguments)
