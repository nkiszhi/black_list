#!/bin/bash

files=$(ls /root/get_blacklist/feeds)

  for element in $files
      do
          echo "$element"
          if [ "${element##*.}"x = "csv"x ]
          then

                 rm -rf  $element
          fi

      done


