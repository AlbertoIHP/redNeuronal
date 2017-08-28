import numpy as np

class RedNeuronal(object):
	
	def __init__(self):
		#Definimos los parametros generales
		
		#Capa entrada
		self.numeroNeuronasEntrada = 4
		
		#Capa escondida
		self.numeroNeuronasEscondidas1 = 6
		
		self.numeroNeuronasEscondidas2 = 6
		
		#Capa salida
		self.numeroNeuronasSalida = 2
		
		#Matrices con los pesos aleatorios
		
		self.W1 = np.random.rand(self.numeroNeuronasEntrada, self.numeroNeuronasEscondidas1)
		
		self.W2 = np.random.rand(self.numeroNeuronasEscondidas1, self.numeroNeuronasEscondidas2)
		
		self.W3 = np.random.rand(self.numeroNeuronasEscondidas2, self.numeroNeuronasSalida)
		
	def avanzar(self, x):
		#Primero calculamos Z2 (paso a)
            
		self.Z2 = np.dot(x, self.W1)
		
                
		#Calculamos las activaciones de la capa 2 (paso b)
		self.A2 = self.sigmoide(self.Z2)
		
		#Calculamos Z3 (paso c)
		self.Z3 = np.dot(self.A2, self.W2)
		
		#Calculamos las activaciones de la capa 3 (paso d)
		self.A3 = self.sigmoide(self.Z3)
		
		#Calculamos Z4 (paso e)
		self.Z4 = np.dot(self.A3, self.W3)
		
		#Calculamos el valor de Y*
		ySombrero = self.sigmoide(self.Z4)
		
		return ySombrero
		
	def sigmoide(self, z):
            return 1/(1 + np.exp(-z))
        
        
        def sigmoidePrima(self, z):
            return np.exp(-z)/((1+np.exp(-z))**2)
        
        
        def funcionDeCosto(self, x, y):
            #La funcion de costo esta definida por la formula planteada en el II paso
            self.ySombrero = self.avanzar(x)
            J = 0.5*sum( ( y - self.ySombrero)**2 )
            return J
		
                
                
        def funcionDeCostoPrima(self, x, y):
            self.ySombrero = self.avanzar(x)
            
            
            #Error de cada uno de los datos evaluados
            E = y - self.ySombrero
            
            delta3 = np.multiply(-(E)  ,self.sigmoidePrima(self.Z4) )
            
            djdw3 = np.dot(self.A3.T, delta3)
            
            delta2 = np.dot(delta3, self.W3.T) * self.sigmoidePrima(self.Z3)
            
            
            djdw2 = np.dot(self.A2.T, delta2)
            
            delta1 = np.dot(delta2, self.W2.T) * self.sigmoidePrima(self.Z2)
            
            djdw1 = np.dot(x.T, delta1)
            
            return djdw1, djdw2, djdw3
            
        def obtenerParametros(self):
            #Transforma las matrices de pesos a un solo vector
            W1_vector = self.W1.ravel()
            W2_vector = self.W2.ravel()
            W3_vector = self.W3.ravel()
            parametros = np.concatenate((W1_vector, W2_vector, W3_vector))
            return parametros
        
        def setearParametros(self, parametros):
            #El vector obtenido con obtenerParametros, se vuelve a configurar a matrices con el fin de reemplazar los pesos antiguos
            W1_start = 0
            W1_end = self.numero
            
            
        
################################# PRUEBA CONSTRUCTOR #################################		
#Realizamos una instancia de una red neuronal		
redNeuronal = RedNeuronal()

#Comprobamos los valores de los pesos 1
print "Valores de los pesos W1"
raw_input(redNeuronal.W1)

#Comprobamos los valores de los pesos 2
print "Valores de los pesos W2"
raw_input(redNeuronal.W2)

#Comprobamos los valores de los pesos 3
print "Valores de los pesos W3"
raw_input(redNeuronal.W3)

############################### PRUEBA PROPAGATION ###################################

#Horas dormidas, Horas de deporte, Horas de ocio y Horas de pareja
X = np.array(([2,3,4,5], [5,1,2,6], [10,2,1,6], [6,7,2,4]), dtype=float)
#Anos de vida restantes, y numero de amigos 
resultados = np.array(([75,12], [82,16], [93,31], [70,52]), dtype=float)
print "Valores de las entradas X sin normalizar"
raw_input(X)
print "Valores de los resultados esperados sin normalzar"
raw_input(resultados)

#Normalizacion de los valores

#Considerando que el maximo puede ser 24 horas 
X = X/24

#Considerando que el maximo de anos de vida restante son 100 y el maximo de amigos tambien es 100 (Puede ser distinto)
resultados = resultados/100 
print "Valores de las entradas X normalizadas"
raw_input(X)
print "Valores de los resultados esperados normalizados"
raw_input(resultados)

print "La prediccion de la red neuronal es"
raw_input(redNeuronal.avanzar(X))

################################# Cost function #####################################

print "Los valores de la funcion de costo son"
raw_input(redNeuronal.funcionDeCosto(X, resultados))

################################ Backpropagation ####################################

print "Los valores de las derivadas son"
a, b, c = redNeuronal.funcionDeCostoPrima(X, resultados)

print "La primera derivada es"
raw_input(a)

print "La segunda derivada es"
raw_input(b)

print "La tercera derivada es"
raw_input(c)

############################### Test gradient numbers ###############################

print "El vector con todos los pesos es"
raw_input(redNeuronal.obtenerParametros())