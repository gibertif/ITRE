import numpy as np
import argparse as ap

parser = ap.ArgumentParser(description='A script to clean files so that they are ready to be used in reweight')
parser.add_argument('--file','-f',help='file to clean')
parser.add_argument('--start','-s',help='First column to retain')
parser.add_argument('--end','-r',help='First columns to discards')
parser.add_argument('--columns','-c',type=int,nargs='+',help='list of columns to use')
parser.add_argument('--discard','-d',help='discard rows after this steps')
parser.add_argument('--output','-o',help='name of the output file')

args = parser.parse_args()

data = np.loadtxt(args.file)

if args.output:
    file_out = args.output
else:
    file_out = '{}_clean'.format(args.file)

if args.discard:
    top = args.discard
else:
    top = len(data)

print(top)
if args.columns:
    np.savetxt(file_out,data[:top,args.columns])
else:
    np.savetxt(file_out,data[:top,args.start:args.end])
