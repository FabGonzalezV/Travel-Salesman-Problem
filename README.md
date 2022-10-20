# TRAVELING SALESMAN PROBLEM(TSP)

 Traveling Salesman Problem  es un  problema típico de optimización combinatoria clasificado como **NP-hard**. La  solución  al problema consiste en encontrar la ruta óptima para un vendedor quien debe recorrer **n** ciudades sin repetirlas de tal modo que optímese su tiempo y dinero, finalizando su recorrido en la ciudad de origen.  El principal problema es la gran cantidad de recorridos que se pueden generar en una ruta simétrica de **n** ciudades (**(n-1) !/2**) Lo que significa que cuando  crece  el  número  de  variables de decisión  del  problema,  el  número  de  decisiones  factibles  y  el  esfuerzo computacional crecen en forma exponencial [1].

A continuación se listan algunas de las aplicaciones de este problema.
 ### Problema de Scheduling 
 Este problema es realmente complejo de resolver. Se formula de la 			  	siguiente forma: hay T tareas que realizar y m procesadores. Se busca una planificación en m procesadores para T minimizando el tiempo. Si ahondamos un poco más en este tipo de problemas, observamos que existen una gran cantidad de variantes asociadas a si hay orden parcial en T (es decir, algunas actividades son preferentes a otras) [2].

###   Problema de placa de circuitos impresos PCB

Ésta es sin duda una de las utilidades más ingeniosas que puede plantear el problema del viajante de comercio: la creación de placas de circuitos. Este problema se enfoca en dos subproblemas: el orden óptimo de taladrar las placas y los caminos óptimos que comunican los chips. En los  **problemas de perforado**  hemos de tomar las ciudades como las posiciones a perforar y las distancias entre ellas como el tiempo que necesita la máquina en trasladarse de una otra. El punto inicial y final será un punto adicional donde permanece la perforadora mientras descansa. Claramente si estas máquinas no son correctamente programadas el tiempo que tarde en recorrer un orificio u otro puede ser significativo con lo que la producción de placas bajaría en un período de tiempo [2].

### Problema de conexión de chips
 La idea es minimizar la cantidad de cable necesaria para unir todos los puntos de una placa sin que haya interferencias. Como los chips son de pequeño tamaño no se pueden poner más de dos cables en un único pin. Tomando las ciudades como los pins y la cantidad de cable necesaria para unirlos como la distancia, el problema es equivalente al de viajante de comercio[2].
 
### Aplicaciones en internet
 
Supongamos que el viajante de comercio es un bit de datos, y que las ciudades son servidores de Red distribuidos por todo el planeta. Esta variante del problema del viajante de comercio es algo inherente al uso óptimo de una plataforma masiva de distribución como es Internet. No olvidemos que en cada ruta puede haber miles de ciudades en este caso. Es curiosos como para resolver esta variante algunos investigadores se han inspirado en el comportamiento de las hormigas[2].  

### Problema de la red de basuras

El problema de la recolección de residuos puede dividirse en 3 grandes tipos: domiciliaria, comercial e industrial. La recolección domiciliaria consiste en atender fundamentalmente casas particulares. La frecuencia puede variar aunque normalmente las rutas suelen repetirse una vez todos los días. La recolección comercial o industrial se encarga de las tiendas, restaurantes o edificios de cocinas. Los objetivos en este tipo de problemas pueden ser diversos: minimizar el número de camiones, la distancia recorrida... Si queremos minimizar la distancia usaremos el problema del viajante de comercio identificando los contenedores o puntos de recogida como las ciudades a visitar. Estos mismos problemas se puede generalizar a los llamados problemas de vehículos o de reparto. Se usan en las empresas de transportes, en correos [2].
___
# PLANTEAMIENTO MATEMÁTICO DEL PROBLEMA

En Traveling Salesman Problem (TSP) existe un conjunto de n ciudades (nodos), **V = {1, 2, 3,..., n}**, y un conjunto de caminos (arcos) uniendo cada una de las ciudades, así el camino **(i,j) Є A**, **cij** es la “distancia”  (*función  objetivo*)  para  ir  de  la  ciudad  **i**  a  la  ciudad  **j**  (**cij**  no  
necesariamente es igual a **cji**). Un vendedor debe realizar un tour (recorrido) comenzando en una cierta ciudad de origen y luego visitar todas las otras ciudades una única vez y retornar a la ciudad de origen. El problema consiste en hallar el tour (recorrido) de distancia mínima evitando subtours. 
El problema del TSP tiene el siguiente modelo matemático[1][9][6][7] . 

​

  $$min \  z (x) = \sum c_{ij}x{ij} (i,j)\in A \\ s.a:\\\sum x_{ij} =1 \ \forall \ j \in V \\\\  \lbrace i:(i,j) \ \in \ A\ \rbrace  \\  \sum x_{ij} =1 \ \forall \ i \in V \\\\  \lbrace j:(i,j) \ \in \ A\ \rbrace  \\  $$
  $$ \sum  x_{ij}\geq 1\ \  \ 2\leq \  \mid U \mid \ \leq \ \mid V \mid - 2\\ \lbrace (i,j)\ \in \ A: \ i \in U, \ j \in (V-U) \rbrace$$

# TÉCNICAS DE SOLUCIÓN
Las técnicas heurísticas son algoritmos que encuentran soluciones de buena  
calidad para problemas combinatoriales complejos; o sea, para problemas tipo  
NP. Los algoritmos heurísticos son más fáciles de implementar y encuentran  
buenas soluciones con esfuerzos computacionales relativamente pequeños, sin embargo, renuncian (desde el punto de vista teórico) a encontrar la solución óptima global de un problema. En problemas de gran tamaño rara vez un algoritmo heurístico encuentra la solución óptima global[1].

Los algoritmos heurísticos se clasifican en algoritmos constructivos (golosos),  
algoritmos de descomposición y división, algoritmos de reducción, algoritmos de manipulación del modelo y algoritmos de búsqueda usando vecindad. En esta última  categoría  pueden  ser  agrupados  los  Algoritmos  Genéticos  (AG), Simulated Annealing (SA), Búsqueda Tabú (TS), Colonia de Hormigas (ACO) y  GRASP[1].

## Algoritmos genéticos
Son herramientas matemáticas que imitan a la naturaleza e intentan resolver  
problemas complejos empleando el concepto de la evolución. El algoritmo  
ejecuta una búsqueda simultánea en diferentes regiones del espacio factible,  
realiza una intensificación sobre algunas de ellas y explora otros subespacios a través de un intercambio de información entre configuraciones[1]. 
Emplea tres  mecanismos básicos que son: La **selección**, el **crossover** y la **mutación**[1][3][4][5][6][7][8][9].
 
 ### Selección: 
  Es  el  operador genético que  permite elegir  las configuraciones  de  la  población  actual  que  deben  participar  de  la  generación  de  las  configuraciones  de  la  nueva  población  (nueva  generación).  Este  operador  termina  después de  decidir  el número  de  descendientes que debe tener cada configuración de la población actual [8].
### Crossover o “Recombinación”:  
Es  el  mecanismo  que  permite  pasar  información  genética  de  un  par  de  
cromosomas originales a sus descendientes, o sea, saltar de un espacio de  
búsqueda a otro, generando de esta forma diversidad genética[8].
### Mutación:
Permite realizar la intensificación en un espacio en particular caminando a  
través de vecinos. Significa intercambiar el valor de un gene de un cromosoma  
en una población. En forma aleatoria, se elige un cromosoma como candidato,  
se genera un número aleatorio y si es menor que la tasa de mutación (ρ<ρm),  
entonces se realiza la mutación. La tasa de mutación se elige del rango [0.001,  
0.05], [8].

 ___ 
En resumen, un AG consiste de los siguientes pasos.

**Inicialización**: se genera  aleatoriamente una población inicial constituida por posibles soluciones del  problema, también llamados individuos. 
 
 **Evaluación**: aplicación de la función de  evaluación a cada uno de los individuos. 
 
**Evolución**: aplicación de operadores  genéticos (como son selección, reproducción y mutación). 
 
**Finalización**:   El AG se deberá detener cuando se alcance la solución óptima, por lo general  ésta se desconoce, así que se deben utilizar otros criterios de detención.  Normalmente se usan dos criterios: 1) correr el AG un número máximo de  iteraciones  (generaciones),  y  2)  detenerlo  cuando  no  haya  cambios  en  la  población. Mientras no se cumpla la condición de término se repite el ciclo:  
***Selección (Se) → Cruzamiento (Cr) → Mutación (Mu) → Evaluación (f(x)) →  
Reemplazo (Re).***  

# Propuesta
## Algoritmo genético para 5 ciudades 
### Algoritmo 
1. Inicialice la población aleatoriamente. 
2. Determinar la aptitud del cromosoma. 
3.  Hasta que termine, repita: 
		 1. Seleccione a los padres. 
		 2. Realizar cruce y mutación. 
		 3. Calcular la aptitud de la nueva población. 
		 4. Añádelo al acervo genético.

### pseudo-código
 Inicializar procedimiento GA 
 { 
 Establecer parámetro de refrigeración = 0; 
 Evaluar la población P(t); 
 Mientras (no hecho) 
 { 
 Padres(t) = Seleccionar_Padres(P(t)); 
 Descendencia(t) = Procrear(P(t)); p(t+1) = Seleccionar_Supervivientes(P(t), Descendencia(t)); 
 t = t + 1;
	    }
   }
   
 ## Código
 

`from random import randint`
 `INT_MAX = 2147483647 `
`V = 5`

`Identificadores de las ciudades`
 `GENES = "ABCDE" `
  `#nodo inicial `
   `START = 0" `
   `#poblacion inicial`

  `POP_SIZE = 10 `

	class individual:
	def __init__(self) -> None:
		self.gnome = ""
		self.fitness = 0

	def __lt__(self, other):
		return self.fitness < other.fitness

	def __gt__(self, other):
		return self.fitness > other.fitness

	def rand_num(start, end):
	return randint(start, end-1)


	 #verifica las ocurrencias de identificadores
	def repeat(s, ch):
	for i in range(len(s)):
		if s[i] == ch:
			return True

	return False

	#muta un gen
	def mutatedGene(gnome):
	gnome = list(gnome)
	while True:
		r = rand_num(1, V)
		r1 = rand_num(1, V)
		if r1 != r:
			temp = gnome[r]
			gnome[r] = gnome[r1]
			gnome[r1] = temp
			break
	return ''.join(gnome)


	 #retorna un gen valido
	def create_gnome():
	gnome = "0"
	while True:
		if len(gnome) == V:
			gnome += gnome[0]
			break

		temp = rand_num(1, V)
		if not repeat(gnome, chr(temp + 48)):
			gnome += chr(temp + 48)

	return gnome


	 #retorna gen valido, la ruta optima
	def cal_fitness(gnome):
	mp = [
		[0, 2, INT_MAX, 12, 5],
		[2, 0, 4, 8, INT_MAX],
		[INT_MAX, 4, 0, 3, 3],
		[12, 8, 3, 0, 10],
		[5, INT_MAX, 3, 10, 0],
	]
	f = 0
	for i in range(len(gnome) - 1):
		if mp[ord(gnome[i]) - 48][ord(gnome[i + 1]) - 48] == INT_MAX:
			return INT_MAX
		f += mp[ord(gnome[i]) - 48][ord(gnome[i + 1]) - 48]

	return f


 
	def cooldown(temp):
	return (90 * temp) / 100
 
	def TSPUtil(mp):
	# numero de generacion
	gen = 1
	# iteraciones de genes
	gen_thres = 5

	population = []
	temp = individual()

	# creando poblacion de genes
	for i in range(POP_SIZE):
		temp.gnome = create_gnome()
		temp.fitness = cal_fitness(temp.gnome)
		population.append(temp)

	print("\n poblacion inicial: \nGNOME	exactitud\n")
	for i in range(POP_SIZE):
		print(population[i].gnome, population[i].fitness)
	print()

	found = False
	temperature = 10000

	# numero de iteraciones
	# cruce de poblaciones y mutación de genes.
	while temperature > 1000 and gen <= gen_thres:
		population.sort()
		print("\nTemperatura actual: ", temperature)
		new_population = []

		for i in range(POP_SIZE):
			p1 = population[i]

			while True:
				new_g = mutatedGene(p1.gnome)
				new_gnome = individual()
				new_gnome.gnome = new_g
				new_gnome.fitness = cal_fitness(new_gnome.gnome)

				if new_gnome.fitness <= population[i].fitness:
					new_population.append(new_gnome)
					break

				else:

					# Aceptar los nodos rechazados en
					# por encima del umbral.
 
					prob = pow(
						2.7,
						-1
						* (
							(float)(new_gnome.fitness - population[i].fitness)
							/ temperature
						),
					)
					if prob > 0.5:
						new_population.append(new_gnome)
						break

		temperature = cooldown(temperature)
		population = new_population
		print("Generacion", gen)
		print("gen -- ruta")

		for i in range(POP_SIZE):
			print(population[i].gnome, population[i].fitness)
		gen += 1


	if __name__ == "__main__":

	mp = [
		[0, 2, INT_MAX, 12, 5],
		[2, 0, 4, 8, INT_MAX],
		[INT_MAX, 4, 0, 3, 3],
		[12, 8, 3, 0, 10],
		[5, INT_MAX, 3, 10, 0],
	]
	TSPUtil(mp)

$$ \\\ \\\ \\\ \\\ \\\ \\\ \\\ \\\ \\\ \\\ \\\ \\\ \\\ \\\ \\\ \\\ \\\ \\\ \\\ \\\ \\\ \\\ \\\ \\\ \\\ \\\ \\\ \\\ \\\ \\\ \\\ \\\ \\\ $$

 # Referencias 

>[1] Noraini Mohd Razali, John Geraghty. (2011). Genetic Algorithm Performance with Different Selection Strategies in Solving TSP. _Proceedings of the World Congress on Engineering_, _II_, Artículo WCE 2011. [http://www.iaeng.org/publication/WCE2011/WCE2011_pp1134-1139.pdf](http://www.iaeng.org/publication/WCE2011/WCE2011_pp1134-1139.pdf)

>[2] TSP: Aplicaciones_. (s. f.). knuth.uca.es. [https://knuth.uca.es/moodle/mod/page/view.php?id=3415](https://knuth.uca.es/moodle/mod/page/view.php?id=3415)

>[3] _Metamodeling the traveling salesman problem in delivery planning_. (s. f.). Séneca Principal. [https://repositorio.uniandes.edu.co/handle/1992/50616](https://repositorio.uniandes.edu.co/handle/1992/50616)

>[4]_Essays in logistics optimization : algorithms and game theory for solving the traveling salesman problem in dynamic scenarios_. (s. f.). Repositorio Institucional da Universidade de Vigo. [http://www.investigo.biblioteca.uvigo.es/xmlui/handle/11093/1172](http://www.investigo.biblioteca.uvigo.es/xmlui/handle/11093/1172)

>[5](s. f.-b). DEPI ITLP. [http://posgrado.lapaz.tecnm.mx/uploads/archivos/2017-1.pdf](http://posgrado.lapaz.tecnm.mx/uploads/archivos/2017-1.pdf)

>[6]_Heuristic Function to Solve The Generalized Covering TSP with Artificial Intelligence Search_. (s. f.). IEEE Xplore. [https://ieeexplore.ieee.org/abstract/document/9281156](https://ieeexplore.ieee.org/abstract/document/9281156)

>[7]_Integración de algoritmos genéticos y redes de Petri como propuesta metodológica para la solución del time-dependent traveling Salesm problem._ (s. f.). Repositorio Digital Univalle. [https://bibliotecadigital.univalle.edu.co/handle/10893/21143](https://bibliotecadigital.univalle.edu.co/handle/10893/21143)

 >[8]_EL Problema del Viajante, heurísticas basadas en algoritmos genéticos The Traveling Salesman Problem, genetic algorithm-based heuristics - E-Prints Complutense_. (s. f.). Archivo Institucional E-Prints Complutense - E-Prints Complutense. [https://eprints.ucm.es/id/eprint/67161/](https://eprints.ucm.es/id/eprint/67161/)
 
 >[9]_Algoritmo de optimización de colonia de hormigas aplicado a TSP, una revisión sistemática - ProQuest_. (s. f.). ProQuest | Better research, better learning, better insights. [https://www.proquest.com/openview/43a681c6eced2135978ec1c925011c0d/1?pq-origsite=gscholar&amp;cbl=1006393](https://www.proquest.com/openview/43a681c6eced2135978ec1c925011c0d/1?pq-origsite=gscholar&amp;cbl=1006393)
