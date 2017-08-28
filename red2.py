import numpy as np
from scipy import optimize

class Nueronal_Network(object):

	def __init__(self, Lambda=0):
		#Definimos los parametros generales
		self.numeroNeuronasEntrada = 2
		self.numeroNeuronasSalida = 1
		self.numeroNeuronasEscondidas = 3
		
		#Se definen los pesos de manera aleatoria, los pesos w1 estan entre la capa de entrada y la escondida
		#con rand es posible generar una matriz de valores aleatorios entre 0 y 1 de tamano especificado en los parametros
		#primer parametro filas, segundo parametro columnas
		self.W1 = np.random.rand(self.numeroNeuronasEntrada, self.numeroNeuronasEscondidas)
		
		#Los pesos w2 estan entre la capa escondida y la de salida
		self.W2 = np.random.rand(self.numeroNeuronasEscondidas, self.numeroNeuronasSalida)
		
		#Agregamos el parametro de umbral o regularizacion
		self.Lambda = Lambda
		
		
	def avanzar(self, x):
	
		#Entregaremos muchos parametros en forma de matriz, dot permite multiplicar esas matrices
		
		self.z2 = np.dot(x, self.W1)
		
		self.a2 = self.sigmoid(self.z2)
		
		self.z3 = np.dot(self.a2, self.W2)
		
		ySombrero = self.sigmoid(self.z3)
		
		return ySombrero
			
	def sigmoid(self, z):
		
		#Aplica la funcion sigmoide sobre una matriz
		return 1/(1+np.exp(-z))
		
	#Es necesario minimizar el costo del error provocado por los pesos, para esto sera necesario corregirlos mediante el descenso del gradiente,
	#para esto, es necesario derivar el costo con respecto al peso lo que nos dara el minimo, se calculara de manera separada para los pesos W1 y W2
	def sigmoidPrima(self, z):
		#Aplica la funcion sigmoide derivada sobre una matriz
		return np.exp(-z)/((1+np.exp(-z))**2)
	
	
	def FuncionDeCosto(self, X, y):
		self.yHat = self.avanzar(X)
		#J = 0.5*sum((y-self.yHat)**2)
		
		
		#We don't want cost to increase with the number of examples, so normalize by dividing the error term by number of examples(X.shape[0])
		J = 0.5*sum((y-self.yHat)**2)/X.shape[0] + (self.Lambda/2)*(np.sum(self.W1**2)+np.sum(self.W2**2))
		return J
		
	
	
	def FuncionDeCostoPrima(self, x, y):
		
		#Obtendremos el valor de y sombrero o asterisco
		self.yHat = self.avanzar(x)
		
		#Posteriormente obtendremos el valor de delta definido como 
		# delta = -(y - y*)* F'(z)
		delta3 = np.multiply(-(y - self.yHat), self.sigmoidPrima(self.z3) )
		
		#Por ultimo multiplicamos la matriz de activaciones traspuesta para que concuerden las dimensionalidades de las matrices
		#Ya que la matriz delta 3 es una matriz de 3 filas y 1 columna que representa los valores multiplicados de los resultados tanto esperados como resueltos por el algoritmo
		
		#Agregamos un valor de umbral Lambda, con el fin de corregir el overfitting de la red tanto
		#para el calculo de la derivada parcial con respecto a los pesos w2 como los w1
		djdw2 = np.dot(self.a2.T, delta3)/x.shape[0] + self.Lambda*self.W2
		
		#Hacemos el calculo del segundo paso del backpropagation
		
		delta2 = np.dot(delta3, self.W2.T)*self.sigmoidPrima(self.z2)
		
		
		djdw1 = np.dot(x.T, delta2)/x.shape[0] + self.Lambda*self.W1
		
		return djdw1, djdw2
		
	def getParams(self):
		#Transforma las matrices de pesos a un solo vector, la funcion ravel transforma una matriz a un vector y la funcion concatenate mezcla dos matrices o vectores en uno solo
		params = np.concatenate((self.W1.ravel(), self.W2.ravel()))
		return params
		
		

	
	def setParams(self, params):
		
		#Define el comienzo de los valroes de los pesos 1
		W1_start = 0
		
		#Define el final de los valores de los pesos 1, obtenidos mediante la multiplicacion del numoro de neuronas en la capa anterior (Entrada) con los
		#de la capa posterior (salida)
		W1_end = self.numeroNeuronasEntrada * self.numeroNeuronasEscondidas
		
		
		#Se re define ahora los pesos 1 con una nueva forma, desde 0 hasta donde terminan los elementos
		#de los pesos 1 en el nuevo vector (Que contiene ambos pesos)
		self.W1 = np.reshape(params[W1_start:W1_end], (self.numeroNeuronasEntrada , self.numeroNeuronasEscondidas))
		
		#Y se calcula el final de los valores de los pesos 2, que sera desde donde terminaron los del peso 1, hasta el final del del vector (Que se puede obtener con un len)
		#o bien multiplicando los valores de la capa anterior (escondida) con los de la posterior (salida)
		W2_end = W1_end + self.numeroNeuronasEscondidas*self.numeroNeuronasSalida
		
		#Se re define ahora los pesos 2 con una nueva forma desde la posicion donde terminaron los pesos anteriores, hasta el final del vector nuevo (params)
		self.W2 = np.reshape(params[W1_end:W2_end], (self.numeroNeuronasEscondidas, self.numeroNeuronasSalida))
		
	def computeGradients(self, X, y):
		#Se obtienen las derivadas mediante la funcion de costo 
		djdw1, djdw2 = self.FuncionDeCostoPrima(X, y)
		#Se devuelven los valores de peso, minimizados en un solo 
		return np.concatenate((djdw1.ravel(), djdw2.ravel()))
		
	def computeNumericalGradient(N, X, y):
		#Este metodo recibe en su primer parametro una red neuronal, mediante la cual obtiene sus parametros en un solo vector
		#Que contiene todos los pesos (tanto W1 como W2)
		paramsInitial = N.getParams()
		
		#Mediante el metodo zeros de numpy, podemos generar una matriz o vector de ceros
		#indicando en el parametro cuantas filas y columnas deseamos en este nuevo elementos
		#En este caso se entrega mediante el metodo shape de numpy, que devuelve las dimensiones
		#que tiene el elemento vectorial (matriz o vector)
		numgrad = np.zeros(paramsInitial.shape)
		perturb = np.zeros(paramsInitial.shape)
		
		#Se declara epsilon en la variable e
		e = 1e-4
		
		#Este ciclo recorre todos los pesos encontrados en el arreglo params que contiene tanto pesos W1 como W2
		for p in range(len(paramsInitial)):
			
			#Decimos que la matriz o vector generado anteriormente (perturb) a medida que avance el ciclo
			#se le introducira en esa posicion el valor epsilon (Recordar que tanto perturb como numgrad tienen la misma dimension que params)
			perturb[p] = e
			
			#Mediante la red neuronal ejecutamos set params y le entregamos todos los parametros (Tanto los pesos W1 como W2) mas la nueva perturbacion
			N.setParams(paramsInitial + perturb)
			
			#Mediante la red neuronal ejecutamos el costo de la funcion definido anteriormente, y lo guardamos como una perdido 2
			loss2 = N.FuncionDeCosto(X, y)
			
			#Repetimos el proceso pero esta vez un poco mas a la izquierda del punto en la grafica (osea restamos)
			N.setParams(paramsInitial - perturb)
			loss1 = N.FuncionDeCosto(X, y)
			
			#Una vez generada la diferencia entonces es posible realizar el calculo de la derivada de la funcion evaluada
			#mediante la diferencia de de las perdidas (Corridas en Epsilon positivo y negativo) dividod en dos veces epsilon
			#Y este resultado es guardado en la nueva matriz o vector generado anteriormente
			numgrad[p] = (loss2 - loss1) / (2*e)
			
			# Regresamos a 0 el valor de la matriz de perturbacion
			perturb[p] = 0
			
		#Finalmente re configuramos ahora a valores aproximados los nuevos pesos generados por el algoritmo
		N.setParams(paramsInitial)
		
		#Retornamos el numero del gradiente
		return numgrad
		
class trainer(object):
		
		def __init__(self, N):
			#Crea una referencia local a la red neuronal
			self.N = N
			
		def costFunctionWrapper(self, params, X, y):
			
			self.N.setParams(params)
			cost = self.N.FuncionDeCosto(X, y)
			grad = self.N.computeGradients(X, y)
			return cost, grad
			
		def callbackF(self, params):
			self.N.setParams(params)
			self.J.append(self.N.FuncionDeCosto(self.X, self.y))
            self.testJ.append(self.N.FuncionDeCosto(self.testX, self.testY))
			
		def train(self, trainX, trainY, testX, testY):
			
			#Guardamos los valroes de entrada y esperados en referencias locales de la clase
			self.X = trainX
			self.y = trainY
			
			#Guardamos los valores de entrada y esperados de prueba 
			self.testX = testX
			self.testY = testY
			
			#Creamos una lista local con el fin de guardar los costos calculados
			self.J = []
			self.testJ = []
			
			
			#Obtenemos la lista con todos los pesos de la red
			params0 = self.N.getParams()
			
			#Definimos un limite de maximas iteraciones (200)
			options = {'maxiter' : 200, 'disp' : True}
			
			_res = optimize.minimize(self.costFunctionWrapper, params0, jac=True, method='BFGS', \
                                 args=(trainX, trainY), options=options, callback=self.callbackF)

			self.N.setParams(_res.x)
			self.optimizationResults = _res


			

trainX = np.array(([3,5], [5,1], [10,2], [6,1.5]), dtype=float)
trainY = np.array(([75], [82], [93], [70]), dtype=float)

#Testing Data:
testX = np.array(([4, 5.5], [4.5,1], [9,2.5], [6, 2]), dtype=float)
testY = np.array(([70], [89], [85], [75]), dtype=float)

#Normalize:
trainX = trainX/np.amax(trainX, axis=0)
trainY = trainY/100 #Max test score is 100

#Normalize by max of training data:
testX = testX/np.amax(trainX, axis=0)
testY = testY/100 #Max test score is 100
	
# X = (hours sleeping, hours studying), y = Score on test
#X = np.array(([3,5], [5,1], [10,2]), dtype=float)
#y = np.array(([75], [82], [93]), dtype=float)

# Normalize
#X = X/np.amax(X, axis=0)
#y = y/100 #Max test score is 100	

redNeuronal = Nueronal_Network(Lambda=0.0001)

print "En base a "
print trainX
print "Los resultados del test respectivamente sera"
raw_input(redNeuronal.avanzar(trainX))

print "En base a "
print testX
print "Los resultados del test respectivamente sera"
raw_input(redNeuronal.avanzar(testX))


print "Entrenando red"
entrenador = trainer(redNeuronal)
entrenador.train(trainX, trainY, testX, testY)

print "En base a "
print trainX
print "Los resultados del test respectivamente sera"
raw_input(redNeuronal.avanzar(trainX))

print "En base a "
print testX
print "Los resultados del test respectivamente sera"
raw_input(redNeuronal.avanzar(testX))
	
			
			
	