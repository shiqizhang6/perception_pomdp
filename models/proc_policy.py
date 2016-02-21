#!/usr/bin/python

import sys

if len(sys.argv) != 2:
    print('wrong numebr of arguments')

f = open(str(sys.argv[1]), 'r')
print('processing ', str(sys.argv[1]))

s = f.read()

start = s.find('<Vector')
end = s.find('/AlphaVector')
s = s[start:end-2]
s = s.replace('<Vector action="', '')
s = s.replace('" obsValue="', ' ')
s = s.replace('">', ' ')
s = s.replace(' </Vector>', ' ')

f = open('processed.policy', 'w')

f.write(s)
