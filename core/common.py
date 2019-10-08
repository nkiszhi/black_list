#!/usr/bin/env python

"""
Copyright (c) 2014-2016 Miroslav Stampar (@stamparm)
See the file 'LICENSE' for copying permission
"""

import csv
import gzip
import os
import re

import StringIO
import urllib2
import zipfile
import zlib

TIMEOUT = 30

def retrieve_content(url, data=None):
    """
    Retrieves page content from given URL
    """

    try:
        req = urllib2.Request("".join(url[i].replace(' ', "%20") if i > url.find('?') else url[i] for i in xrange(len(url))), data, {"User-agent": "nkamg-trail", "Accept-encoding": "gzip, deflate"})
        resp = urllib2.urlopen(req, timeout=TIMEOUT)
        retval = resp.read()
        encoding = resp.headers.get("Content-Encoding")

        if encoding:
            if encoding.lower() == "deflate":
                data = StringIO.StringIO(zlib.decompress(retval, -15))
            else:
                data = gzip.GzipFile("", "rb", 9, StringIO.StringIO(retval))
            retval = data.read()
    except Exception, ex:
        retval = ex.read() if hasattr(ex, "read") else getattr(ex, "msg", str())

    return retval or ""






