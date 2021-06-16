@echo off
D:
cd D:\Anaconda\Spyder workspace\assignment2
echo BINARY

python IR_engine.py -o results.txt -w binary -s -p
echo With stoplist and stemming
python eval_ir.py -F cacm_gold_std.txt results.txt

python IR_engine.py -o results.txt -w binary -s
echo With stoplist
python eval_ir.py -F cacm_gold_std.txt results.txt 

python IR_engine.py -o results.txt -w binary -p
echo With stemming
python eval_ir.py -F cacm_gold_std.txt results.txt 

python IR_engine.py -o results.txt -w binary
echo Without stoplist and stemming
python eval_ir.py -F cacm_gold_std.txt results.txt 
echo ---------------------------

echo TF
python IR_engine.py -o results.txt -w tf -s -p
echo With stoplist and stemming
python eval_ir.py -F cacm_gold_std.txt results.txt

python IR_engine.py -o results.txt -w tf -s
echo With stoplist
python eval_ir.py -F cacm_gold_std.txt results.txt 

python IR_engine.py -o results.txt -w tf -p
echo With stemming
python eval_ir.py -F cacm_gold_std.txt results.txt 

python IR_engine.py -o results.txt -w tf
echo Without stoplist and stemming
python eval_ir.py -F cacm_gold_std.txt results.txt 
echo ---------------------------

echo TFIDF
python IR_engine.py -o results.txt -w tfidf -s -p
echo With stoplist and stemming
python eval_ir.py -F cacm_gold_std.txt results.txt

python IR_engine.py -o results.txt -w tfidf -s
echo With stoplist
python eval_ir.py -F cacm_gold_std.txt results.txt 

python IR_engine.py -o results.txt -w tfidf -p
echo With stemming
python eval_ir.py -F cacm_gold_std.txt results.txt 

python IR_engine.py -o results.txt -w tfidf
echo Without stoplist and stemming
python eval_ir.py -F cacm_gold_std.txt results.txt 
echo ---------------------------
pause
