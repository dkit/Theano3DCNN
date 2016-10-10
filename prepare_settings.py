from data_handling import create_training_settings, create_testing_settings, dump_settings_file

create_training_settings('data', ['boxing/training', 'handclapping/training', 
                                  'handwaving/training', 'jogging/training', 
                                  'running/training', 'walking/training'])
dump_settings_file('data/training_data_settings.p')										  
create_testing_settings('/data', ['boxing/testing', 'handclapping/testing', 
								  'handwaving/testing', 'jogging/testing', 
								  'running/testing', 'walking/testing'])
dump_settings_file('data/testing_data_settings.p')					  

