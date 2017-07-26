#!/usr/bin/env python
__author__ = 'jesse'

import argparse
import os
import pickle


def main():

    features_fn = FLAGS_features_fn
    outdir = FLAGS_outdir

    # Read in features pickle
    print "reading in features pickle and extracting meta-data..."
    with open(features_fn, 'rb') as f:
        features = pickle.load(f)

    # Get object and context information from features structure.
    nb_objects = len(features)
    contexts = []
    for b in features[0].keys():
        for m in features[0][b].keys():
            c = (b, m)
            if c not in contexts:
                contexts.append(c)
    print "... done"

    # Write context files to out directory.
    print "writing out context files to '" + outdir + "'"
    for b, m in contexts:
        fn = b + "_" + m.replace('_', '-') + ".txt"
        with open(os.path.join(outdir, fn), 'w') as f:
            lines = []
            for oidx in range(nb_objects):
                for obsidx in range(len(features[oidx][b][m])):
                    lines.append(','.join([str(oidx + 1), str(obsidx + 1)] +
                                          [str(d) for d in features[oidx][b][m][obsidx]]))
            f.write('\n'.join(lines))
    print "... done"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--features_fn', type=str, required=True,
                        help="features pickle file")
    parser.add_argument('--outdir', type=str, required=True,
                        help="directory to output feature text files")
    args = parser.parse_args()
    for k, v in vars(args).items():
        globals()['FLAGS_%s' % k] = v
    main()
