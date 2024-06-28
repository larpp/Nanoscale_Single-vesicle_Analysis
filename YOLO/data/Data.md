~~~
📦data
 ┣ 📂train
 ┃ ┣ 📂images
 ┃ ┃ ┣ 📜train.jpg
 ┃ ┗ 📂labels
 ┃ ┃ ┣ 📜train.txt
 ┣ 📂valid
 ┃ ┣ 📂images
 ┃ ┃ ┣ 📜valid.jpg
 ┃ ┗ 📂labels
 ┃ ┃ ┣ 📜valid.txt
 ┣ 📂test
 ┃ ┣ 📂images
 ┃ ┃ ┣ 📜test.jpg
 ┃ ┗ 📂labels
 ┃ ┃ ┣ 📜test.txt
 ┣ 📜.gitkeep
~~~

---

#### ex ) labels/train.txt 
~~~md 
# ClassID x_center y_center Width Height
0 0.53828125 0.0796875 0.01875 0.015625
0 0.6421875 0.34296875 0.0171875 0.01875
0 0.465625 0.53125 0.015625 0.015625
0 0.79140625 0.0703125 0.015625 0.0171875
0 0.4734375 0.32734375 0.0125 0.00625
0 0.67109375 0.9921875 0.0140625 0.015625
0 0.21953125 0.784375 0.0140625 0.0125
0 0.55 0.27734375 0.0125 0.015625
0 0.97421875 0.26171875 0.009375 0.0125
~~~
- (x_center y_center Width Height) : 0 ~ 1 Normalize

---
