# save numpy array as npy file
from numpy import asarray
from numpy import save
# define data
data = asarray([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
# save to npy file
save('data.npy', data)
