#!/usr/bin/env python2


def deduplicate(exe_file):
    lines = []
    with open(exe_file) as f:
        l = f.readline()
        if l not in lines:
            lines.append(l)

    with open(exe_file, "w") as f:
        f.write('\n'.join(lines))
