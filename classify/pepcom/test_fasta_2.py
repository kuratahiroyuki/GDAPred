import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--test_file', help='file')
parser.add_argument('--gen_file', help='file')
parser.add_argument('--test_fa', help='file')
parser.add_argument('--test_csv', help='file')
parser.add_argument('--gen_test_fa', help='file')
parser.add_argument('--gen_test_csv', help='file')

test_file = parser.parse_args().test_file
gen_file = parser.parse_args().gen_file
test_fa = parser.parse_args().test_fa
test_csv = parser.parse_args().test_csv
gen_test_fa = parser.parse_args().gen_test_fa
gen_test_csv = parser.parse_args().gen_test_csv

test = pd.read_csv(test_file , header=None)
num_gen = test[test[1]==1].shape[0]
gen = pd.read_csv(gen_file, header=None)
test = pd.concat([gen.iloc[:num_gen], test[test[1]==0]])
test = test.rename(columns={0:'seq',1:'label'})
print(f"{test=}")
gen_test = gen.rename(columns={0:'seq',1:'label'}) #1000
print(f"{gen_test=}")

with open(test_fa, 'w') as fout:
   for i in range(test.shape[0]):
      if test.iloc[i,1] == 1:
         fout.write('>pep_%s|1|label\n'%i)
         fout.write(test.iloc[i,0])
         fout.write('\n')
      else:
         fout.write('>pep_%s|0|label\n'%i)
         fout.write(test.iloc[i,0])
         fout.write('\n')

test.to_csv(test_csv, index=None)
gen_test.to_csv(gen_test_csv, index=None)
    
