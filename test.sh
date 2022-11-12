#!/bin/sh
set -x

usage() { echo "Usage: $0 -t <test.bin> [-g]" 1>&2; exit 1; }

while getopts ":rmgt:" o; do
    case "${o}" in
        t)
            TESTBIN=${OPTARG}
            ;;
        g)
            NOGEN=1
            ;;
        *)
            usage
            ;;
    esac
done
shift $((OPTIND-1))

if [ -z "${TESTBIN}" ]; then
    usage
fi

if [ -z "${NOGEN}" ]; then
./gen.exe -s 128 -l 100
./gen.exe -s 256 -l 200
./gen.exe -s 512 -l 400
./gen.exe -s 1024 -l 800
./gen.exe -s 1536 -l 1600
fi

./${TESTBIN} -k key.txt -i gen_0128mb_1.txt -o m1.enc
./${TESTBIN} -k key.txt -i gen_0256mb_1.txt -o m2.enc
./${TESTBIN} -k key.txt -i gen_0512mb_1.txt -o m3.enc
./${TESTBIN} -k key.txt -i gen_1024mb_1.txt -o m4.enc
./${TESTBIN} -k key.txt -i gen_1536mb_1.txt -o m5.enc
./${TESTBIN} -k key.txt -o gen_0128mb_2.txt -i m1.enc
RES=`diff gen_0128mb_1.txt gen_0128mb_2.txt`
if [ -z $RES ]; then
    echo "gen_0128mb_1.txt gen_0128mb_2.txt are identical"
else
    echo "files are differ"
fi