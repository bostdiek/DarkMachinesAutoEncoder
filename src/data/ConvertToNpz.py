import numpy as np
import argparse
import pathlib

parser = argparse.ArgumentParser(description='Convert csv to npz file')
parser.add_argument('--input',
                    type=pathlib.Path,
                    required=True,
                    dest='file_in'
                    )
parser.add_argument('--output',
                    type=pathlib.Path,
                    required=True,
                    dest='file_out'
                    )

args = parser.parse_args()
print(args.file_in)

max_size = 0
with open(args.file_in) as f:
    for line in f:
        max_size += 1

reconstructed_objects = np.zeros([max_size, 20*5])
Processes = []
with open(args.file_in) as f:
    for i, line in enumerate(f):
        line = line.strip().split(';')

        Processes.append(line[1])
        reconstructed_objects[i, 0] = 8
        reconstructed_objects[i, 2] = float(line[3])
        reconstructed_objects[i, 4] = float(line[4])

        for j, obj in enumerate(line[5:-1]):
            j=j+1
            obj = obj.split(',')
#             print(obj)
            obj, e, pt, eta, phi = obj
            obj_convert = {'j': 1, 'b':2, 'e-': 3, 'e+': 4, 'm-':5, 'm+':6, 'g': 7}
            reconstructed_objects[i, j*5] = obj_convert[obj]
            reconstructed_objects[i, j*5 + 1] = float(e)
            reconstructed_objects[i, j*5 + 2] = float(pt)
            reconstructed_objects[i, j*5 + 3] = float(eta)
            reconstructed_objects[i, j*5 + 4] = float(phi)


reconstructed_objects = reconstructed_objects.reshape(max_size, 20, 5)
Processes = np.array(Processes)

np.savez(args.file_out,
         Objects=reconstructed_objects,
         Procs=Processes
         )
