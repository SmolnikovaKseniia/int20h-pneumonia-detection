from tensorflow.keras.models import load_model
test_images = '/kaggle/input/rsna-pneumonia-detection-challenge/stage_2_test_images'
test_filenames = os.listdir(test_images)
print('n test samples:', len(test_filenames))

test_gen = Generator(test_images, test_filenames, None, batch_size=25, image_size=256, shuffle=False, predict=True)
submission_dict = {}
model = load_model('my_model.h5')

for imgs, filenames in test_gen:
    preds = model.predict(imgs)
    for pred, filename in zip(preds, filenames):
        pred = resize(pred, (1024, 1024), mode='reflect')
        comp = pred[:, :, 0] > 0.5
        comp = measure.label(comp)
        predictionString = ''
        for region in measure.regionprops(comp):
            y, x, y2, x2 = region.bbox
            height = y2 - y
            width = x2 - x
            conf = np.mean(pred[y:y+height, x:x+width])
            predictionString += str(conf) + ' ' + str(x) + ' ' + str(y) + ' ' + str(width) + ' ' + str(height) + ' '
        filename = filename.split('.')[0]
        submission_dict[filename] = predictionString
      
    if len(submission_dict) >= len(test_filenames):
        break


sub = pd.DataFrame.from_dict(submission_dict,orient='index')
sub.index.names = ['patientId']
sub.columns = ['PredictionString']
sub.to_csv('submission.csv')
