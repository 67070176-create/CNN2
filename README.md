This Used Rasnet18 defualt weight

resolution 244,244

data augmentation
- rotate +-10 
- img shift
- size/scale
- light change(little)
- normalize to standard of rasnet input
- erotion/dialation จำลองตัวบาง/หนา
- randomErasing สุ่มลบส่วนภาพ

hightlight i think is
1. class balance by add weight each class
2. doup_out 30%
3. learning rate with decay
4. early stop
5. save data in train loop using torch so it doing on GPU instead of CPU(lower time lag at communicate) EX.running_loss ,correct
6. calc. val_loss in train loop to see overfit and early stop and save best model
7. have checkpoint to save both model and index(in case test data sort differently)


#### มีการอธิบายชุดข้อมูล อาทิเช่น จำนวนคลาส จำนวนข้อมูลแต่ละคลาส ตัวอย่างข้อมูลในแต่ละคลาส
- 72 classes
- imange in class have size around 10*10 to 35*35
- .jpg
- have over 62,711 picture
#### มีการวิเคราะห์ความท้าทายของชุดข้อมูลนี้
- serve unbalance classes
- low resolution
- simirality in some class
- distort image/character
#### มีการอธิบายถึงโครงสร้าง CNN ที่ใช้งานในภาพรวม
- rasnet18
-input image to 64 7x7 kernel filter stride 2
- pooling 2 stride 2
- use 3x3 conv to increase channel to finetune detail(64>128>256>512)
  - each layer conv 3 timed
  - after 64 layer(start 128) first conv stride 2
  - rasnet connectivity
- 7x7 pooling stride 7
- FC 1000(1000x1x1)
- softmax
##### add for this CNN
- drop out 30%
- linear output to 72 classes instead of 1000

#### มีการอธิบายถึงเทคนิคการถ่ายโอนความรู้ (Transfer Learning)  
- use rasnet18 model and it's defualt weight 
#### มีการใอธิบายถึงเทคนิคการสังเคราะห์ข้อมูล (Data Augmentation) 
- rotate +-10 
- img shift
- size/scale
- light change(little)
- normalize to standard of rasnet input
- erotion/dialation จำลองตัวบาง/หนา
- randomErasing สุ่มลบส่วนภาพ
#### มีการอธิบายถึงเทคนิคหรือแนวคิดที่น่าสนใจ
1. class balance by add weight each class
2. doup_out 30%
3. learning rate with decay
4. early stop
5. save data in train loop using torch so it doing on GPU instead of CPU(lower time lag at communicate) EX.running_loss ,correct
6. calc. val_loss in train loop to see overfit and early stop and save best model
7. have checkpoint to save both model and index(in case test data sort differently)
8. erotion/dialation จำลองตัวบาง/หนา
#### มีการอธิบายถึงการทำงานของ CNN โดยเน้นที่จุดเด่นของสถาปัตยกรรม 
- use 3x3 conv to increase channel to finetune detail(64>128>256>512)
  - each layer conv 3 timed
  - after 64 layer(start 128) first conv stride 2
  - rasnet connectivity(Residual Blocks)
- 7x7 pooling stride 7 (GAP)
- FC 1000(1000x1x1)(fully connected)
#### มีการอธิบายถึงเทคนิคที่ใช้งาน 
- doup_out 30%
- early stop
- learning rate with decay
- save data in train loop using torch so it doing on GPU instead of CPU(lower time lag at communicate) EX.running_loss ,correct
- calc. val_loss in train loop to see overfit and early stop and save best model
-  class balance by add weight each class
#### มีการอธิบายถึงขั้นตอนการฝึกสอนแบบจำลอง
- yea bro
#### มีการแสดงประสิทธิภาพของชุดฝึกสอนด้วยค่าความถูกต้อง (Accuracy Rate)

#### มีการแบ่งชุดข้อมูลออกเป็น Train และ Validation ในสัดส่วน 80% ต่อ 20% 

มีการแนะนำสมาชิกในกลุ่ม
