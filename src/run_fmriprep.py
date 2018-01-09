from subprocess import Popen 
from grabbit import Layout
import numpy as np

# Get subjects
layout = Layout('../data', './grabbit_config.json')
subjects = layout.get(target='subject', modality='func', return_type='id')

# Change these as necessary
# Must be absolute paths
data_path = '/autofs/space/cassia_001/users/matt/msit/data'
fmriprep_path = '%s/derivatives/fmriprep' % data_path
subject_ix = np.arange(50, 59)

# Run each subject in the index list above on a separate process in parallel
for i in subject_ix:

    docker_command = ['docker', 'run', '--rm',
			          '-v', '%s:/data:ro' % data_path,
		              '-v', '%s:/out' % fmriprep_path,
					  '-v', '%s:/work' % fmriprep_path,
					  '-w', '/work',
					  'poldracklab/fmriprep:latest',
					  '/data', '/out',
					  'participant',
					  '--participant_label', subjects[i],
					  '-t', 'msit',
					  '-w', '/work',
				      '--omp-nthreads', '1',
					  '--nthreads', '1',
					  '--output-space', 'fsaverage', 'fsnative', 
					  'T1w', 'template']
    print(' '.join(docker_command))
    Popen(docker_command)
