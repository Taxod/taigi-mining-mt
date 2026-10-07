import argparse
import logging
import numpy as np


def getArguments():
  parser = argparse.ArgumentParser()

  parser.add_argument('-i', '--input', type=str, required=True, help='Format: <SID>\t<TID>\t<similarity>')
  parser.add_argument('-o', '--output', type=str, required=True)
  parser.add_argument('-m', '--method', default='dynamic', type=str, help='Filtering method: static or dynamic')
  parser.add_argument('-th', '--threshold', default=2.0, type=float, help='In case of static method: threshold value; dynamic: weight (lambda) of similarity standard deviation: th = mean(S) + lambda*std(S)')

  return parser.parse_args()


def main(input, output, method='dynamic', threshold=2.0):
    assert method in ['static', 'dynamic']

    if method == 'dynamic':
        # calculating threshold
        tmp_lst = list()
        with open(input, 'r') as fin:
            for line in fin:
                best = float(line.split('\t')[2])
                if not np.isnan(best) and not np.isinf(best):
                    tmp_lst.append(best)
        
        if len(tmp_lst) == 0:
            print('The dynamic threshold value is nan.')
            threshold = float('inf')
        else:
            s = np.array(tmp_lst)
            threshold = s.mean() + threshold*s.std()
            print(f'The dynamic threshold value is {threshold}.')

    # filtering
    with open(input, 'r') as fin, open(output, 'w') as fout:
      for line in fin:
          data = line.split('\t')
          sid = data[0]
          tid = data[1]
          tsim = float(data[2])

          if not np.isnan(tsim) and not np.isinf(tsim) and tsim > threshold:
              fout.write('{}\t{}\n'.format(sid, tid))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    args = getArguments()

    main(**vars(args))
