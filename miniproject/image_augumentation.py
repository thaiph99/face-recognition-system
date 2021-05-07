img = load_img('girl.jpg')
img = img_to_array(img)
data = expand_dims(img, 0)
 
# Dinh nghia 1 doi tuong Data Generator voi bien phap chinh sua anh Zoom tu 0.5x den 2x
myImageGen = ImageDataGenerator(zoom_range=[0.5,2.0])
# Batch_Size= 1 -> Moi lan sinh ra 1 anh
gen = myImageGen.flow(data, batch_size=1)