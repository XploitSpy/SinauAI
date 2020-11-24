# Memasukan library
from numpy import loadtxt
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense

# Masukkan dataset
# Memilih 8 kolom pertama dari indeks 0 - 7
# Memilih kolom output (variabel ke-9) melalui indeks 8
dataset = loadtxt('pima-indians-diabetes.csv', delimiter=',')
X = dataset[:,0:8]
y = dataset[:,8]

# Definisi Model Keras
# Lapisan tersembunyi pertama 10 node dengan aktivasi rule
# Lapisan tersembunyi dua 6 node dengan aktivasi rule
# lapisan keluaran 1 node dengan aktivasi sigmoid 
# Rule berfungsi mengubah nilai negatif menjadi 0
# Sigmoid Berfungsi mengubah nilai aktivasi menjadi nilai antara 0 dan 1
model = Sequential()
model.add(Dense(10, input_dim=8, activation='relu')) 
model.add(Dense(6, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile Model Keras
# membuat argumen kerugian dengan cross entropy
# Mengoptimalkan algoritma gradient descent dengan menggunakan adam
# Membuat keakrutan klasifikasi dengan argumen metric
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Eksekusi Model
# 
# Epoch : Satu melewati semua baris dalam set data pelatihan.
# Batch : Satu atau beberapa sampel yang dipertimbangkan oleh model dalam suatu periode sebelum bobot diperbarui.
# Satu epoch terdiri dari satu atau beberapa batch
model.fit(X, y, epochs=150, batch_size=20)

# Evaluasi Model
# Fungsi eval untuk mengembalikan daftar dengan dua nilai.
# Yang pertama adalah hilangnya model pada dataset dan yang kedua adalah keakuratan model pada dataset.
_, accuracy = model.evaluate(X, y)
print('Accuracy: %.2f' % (accuracy*100))
