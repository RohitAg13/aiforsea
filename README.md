## Grab - challenge
 
How might we automate the process of recognizing the details of the vehicles from images, including make and model?

<B> PROBLEM STATEMENT</B>

Given a dataset of distinct car images, can you automatically recognize the car model and make?

<b>DATASET</b>

Stanford's Car's [Dataset](https://ai.stanford.edu/~jkrause/cars/car_dataset.html)

### How to Test:

1. `git clone https://github.com/RohitAg13/aiforsea.git`

2. `conda install -c pytorch -c fastai fastai` to install pytorch and fastai latest version.

3. Download the weights from [here](https://drive.google.com/file/d/1Yc1cYX05bdWjt5OW538nxoQbamhDTDJi/view?usp=sharing) and keep it inside `data/` folder.

3. keep all the test images in a new folder (eg `test-folder`) and run `python test.py --test <replace-test-folder-name>`  to generate the prediction and store it in `result.csv `.



