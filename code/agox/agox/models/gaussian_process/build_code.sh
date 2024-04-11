DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

cd $DIR/featureCalculators_multi
# rm *.c *.o
python3 setup.py clean
python3 setup.py build_ext --inplace --force

cd $DIR/delta_functions_multi
# rm *.c *.o
python3 setup.py clean
python3 setup.py build_ext --inplace --force