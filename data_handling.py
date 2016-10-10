import glob
from PIL import Image
import theano
import numpy as np
import pymedia.muxer as muxer
import pymedia.video.vcodec as vcodec
from theano.tensor import TensorType
import pygame
import pickle
import os

training_settings_file = 'data/training_data_settings.p'
testing_settings_file = 'data/testing_data_settings.p'

dtype_to_use = 'float32'
int_type = 'int32'

def read_frames_from_video(avi_file, image_size, color_channels=3, count_only=False, read_range = None):
	dm= muxer.Demuxer( avi_file.split( '.' )[ -1 ] )
	f= open( avi_file, 'rb' )
	s= f.read( 400000 )
	r= dm.parse( s )
	v= filter( lambda x: x[ 'type' ]== muxer.CODEC_TYPE_VIDEO, dm.streams )
	if len( v )== 0:
		raise Exception('There is no video stream in a file %s' % inFile)
	v_id= v[ 0 ][ 'index' ]
	c= vcodec.Decoder( dm.streams[ v_id ] )
	frame_count = 0;
	
	if count_only:
		frames = None
	elif read_range == None:
		frames = np.zeros((1, color_channels, image_size[1], image_size[0]), dtype=theano.config.floatX)
	else:
		time_depth = read_range[1]-read_range[0]+1
		frames = np.zeros((time_depth, color_channels, image_size[1], image_size[0]), dtype=theano.config.floatX)
		print(frames.shape)
		
	have_read = False
	finished = False
	actual_frames_read = 0
	while len( s )> 0 and not finished:
		for fr in r:
			if fr[ 0 ]== v_id:
				d= c.decode( fr[ 1 ] )
				# Save file as RGB BMP
				if d:
					if count_only:
						frame_count += 1
						actual_frames_read += 1
						continue;
					if not read_range == None and frame_count < read_range[0]:
						frame_count = frame_count + 1
						continue
					elif not read_range == None and frame_count > read_range[1]:
						finished = True
						break
						
					dd= d.convert( 2 )
					
					if color_channels == 3:
						img = Image.frombytes("RGB", dd.size, dd.data).resize(image_size)
					elif color_channels == 1:
						img = Image.frombytes("RGB", dd.size, dd.data).convert('L').resize(image_size)
					else:
						raise Exception("Unknown color_channels value")
					
					img = np.asarray(img, dtype=dtype_to_use) / 256.
					if (color_channels == 1): img = np.expand_dims(img, 2)
					img = img.transpose(2, 0, 1) #.reshape(1, color_channels, image_size[1], image_size[0])
					if not have_read:
						frames[0, :, :, :] = img
						have_read = True
					elif read_range == None:
						frames = np.concatenate((frames, np.expand_dims(img, 0)), 0)
					else:
						frames[actual_frames_read, :, :, :] = img
					
					frame_count += 1
					actual_frames_read += 1
					
		if not finished:
			s= f.read( 400000 )
			r= dm.parse( s )

	return (actual_frames_read, frames)

def create_settings_file(settings_filename, directory, sets):
	settings = {}
	enumerated = []
	labels = []
	frames = []
	inverse_label_map = {}
	inverse_frame_map = {}
	label_count = 0;
	for k in sets:
		for i in glob.glob(directory + "/" + k + "/*.avi"):
			enumerated.append(i)
			labels.append(label_count)
			inverse_label_map[i] = label_count
			num_frames, ignore = read_frames_from_video(i, (0,0), color_channels=1, count_only=True)
			frames.append(num_frames)
			inverse_frame_map[i] = num_frames
		label_count += 1
			
			
	settings['settings_filename'] = settings_filename
	settings['directory'] = directory
	settings['sets'] = sets
	settings['files'] = enumerated
	settings['labels'] = labels
	settings['inverse_label_map'] = inverse_label_map
	settings['frame_count'] = frames
	settings['inverse_frame_map'] = inverse_frame_map
	pickle.dump( settings, open( settings_filename, "wb" ) )

def dump_settings_file(settings_filename):
	settings = pickle.load( open( settings_filename, "rb" ) )
	print(settings)

def load_training_settings():
	if os.path.isfile(training_settings_file):
		settings= pickle.load( open( training_settings_file, "rb" ) )
	else:
		raise Exception("The training settings file does not exist. Please run the 'prepare_settings.py' script.")	
	return settings

def load_testing_settings():
	if os.path.isfile(testing_settings_file):
		settings= pickle.load( open( testing_settings_file, "rb" ) )
	else:
		raise Exception("The training settings file does not exist. Please run the 'prepare_settings.py' script.")	
	return settings
	
def create_training_settings(directory, sets):
	create_settings_file(training_settings_file, directory, sets)

def create_testing_settings(directory, sets):
	create_settings_file(testing_settings_file, directory, sets)
	
def load_random_data_set(data_settings, batchsize, in_time, in_channels, in_width, in_height, round=None):
	cache_file = 'cache/set%s.p' % round
	if not round == None and os.path.isfile(cache_file):
		dataset = pickle.load( open( cache_file, "rb" ) )
		return dataset
		
	files_to_use = np.random.randint(len(data_settings['files']), size = batchsize)
	
	file_counts = {}
	for i in files_to_use:
		if data_settings['files'][i] in file_counts.keys():
			file_counts[data_settings['files'][i]] += 1
		else:
			file_counts[data_settings['files'][i]] = 1
	
	labels = [-1]*batchsize

	segments = np.zeros((batchsize, in_time, in_channels, in_height, in_width), dtype = TensorType(theano.config.floatX, (False,)*5));

	counter = 0
	for i in file_counts.keys():
		print("Loading %d segments from %s which has %d frames" % (file_counts[i], i, data_settings['inverse_frame_map'][i]))
		frame_numbers = np.random.randint(data_settings['inverse_frame_map'][i]-in_time, size = file_counts[i])
		for j in frame_numbers:
			print("Chose to use frame number %d" % j)
			num_read, video_frames = read_frames_from_video(i, (in_width, in_height), in_channels, read_range = (j, j+in_time-1))
			labels[counter] = data_settings['inverse_label_map'][i]
			segments[counter, :, :, :, :] = video_frames
			counter += 1
	
	if not epoch == None:
		labels = np.asarray(labels,dtype=TensorType(int_type, (False,)))
		dataset = pickle.dump( (labels, segments), open( cache_file, "wb" ) )
	
	return (labels, segments)
	
	