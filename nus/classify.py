from __future__ import print_function
import paho.mqtt.client as mqtt
import numpy as np
from PIL import Image

USERID = "sws007"
PASSWORD = "bengalcat"

TMP_FILE="/tmp/"+USERID+".jpg"
dict = {0:'daisy', 1:'dandelion', 2:'roses',3:'sunflowers', 4:'tulips'}

def load_image(image):
	img = Image.open(image)
	img = img.resize((249,249))
	imgarray = np.array(img)/255.0
	final = np.expand_dims(imgarray, axis=0)
	return final

def classify(imgarray, dict):
	return dict[4],0.98,4

def on_connect(client, userdata, flags, rc):
	print("Connected with result code %d."%(rc))
	
	client.subscribe(USERID +"/IMAGE/classify")
	
def on_message(client, userdata, msg):

	print("Received message. Writing to %s."%(TMP_FILE))
	
	img = msg.payload
	
	with open(TMP_FILE,"wb") as f:
		f.write(img)
		
	imgarray = load_image(TMP_FILE)
	
	label, prob, index = classify(imgarray, dict)
	
	print("Classified as %s with certainty %3.4f."%(label, prob))
	
	client.publish(USERID +"/IMAGE/predict", label + ":" +str(prob)+":"+str(index))
	
def setup():

	global dict
	global client
	
	client = mqtt.Client(transport="websockets")
	
	client.username_pw_set(USERID, PASSWORD)
	client.on_connect = on_connect
	client.on_message = on_message
	
	print("connecting")
	
	x = client.connect("pi.smbox.co", 80,30)
	client.loop_start()
	
def main():
	setup()
	while True:
		pass
		
if __name__ == '__main__':
	main()