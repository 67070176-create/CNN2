This Used Rasnet18 defualt weight

resolution 244,244

data augmentation
- rotate +-10 
- img shift
- size/scale
- light change(little)
- normalize to standard of rasnet input

hightlight i think is
1. class balance by add weight each class
2. doup_out 30%
3. learning rate with decay
4. early stop
5. save data in train loop using torch so it doing on GPU instead of CPU(lower time lag at communicate) EX.running_loss ,correct
6. calc. val_loss in train loop to see overfit and early stop and save best model
7. have checkpoint to save both model and index(in case test data sort differently)
