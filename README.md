# seamcarver_python

how to use:
python run_cmd.py [image path] [forward or backward] [elongate or compress] [target width] [gradient choice] [destination path]

gradient choices: "abs","norm","sobel1_norm","sobel1_abs","sobel3_norm","sobel3_abs"

example:
python run_cmd.py images/fig7.png forward elongate 450 abs images/fig7_test.png

note to save code I only implemented the horizontal dimension change, if you want to do vertical simply rotate your image first and then call the script=)
