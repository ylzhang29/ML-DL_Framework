layer1= 400
layer2= 100
lr = .001
batch_size = 24

optimizer = Adam(lr = lr)
act = 'relu'

input = Input(shape=(len(X_train[1]),))
deep = Dense(units=layer1, activation=act)(input)
bn = Dense(units=layer2, activation=act)(deep)
deep = Dense(units=layer1, activation=act)(bn)
outlayer = Dense(units=len(X_train[1]), activation='linear')(deep)

model = Model(input, outlayer)
encoder = Model(input, bn)

model.compile(loss=['mse'], metrics=['mae'],optimizer=Adam(lr = lr))

model.fit(X_train, X_train,
                    epochs=100,
                    batch_size=batch_size,
                    shuffle=True, verbose=2,
                    validation_data=(X_valid, X_valid))
