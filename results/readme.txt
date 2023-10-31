Un léeme improvisado.

El código que hay aquí es parte del que he utilizado para el trabajo, he seleccionado y reorganizado lo que iba al grano del draft.

En la carpeta 'src' están los siguientes archivos:
	- data_utils.py: Alberga funciones relativa a la descarga, importación, gestión de datos, etc.
	- utils.py: Miscelánea.
	- exps_figures.py: Los experimentos para producir las figuras del draft. Ejecutar directamente para 		producir figuras, no he puesto input por teclado.
	- model_age_sir_vac.py: Modelo age-SIR con vacunación y funciones relacionadas. Se puede
	ejecutar directamente para producir un plot de la dinámica para un estado concreto. Cualquier parámetro a 	cambiar habría que hacerlo internamente, no he puesto tampoco input por teclado.
	- model_age_sir.py: Modelo age-SIR sin vacunación. Ídem.
	- plots.py: Alberga todas las funciones de representación gráfica.

En 'data' estarían todos los ficheros con los datos necesarios para alimentar el modelo.
En 'results' se guardarían los resultados de los plots.


EL PROBLEMA: Como te dije, una vez introduje el factor chi de susceptibilidad en las
ecuaciones, por desconocimiento seguí calculando beta como beta=(R0/gamma)*largest_eigenvalue(contact). Esa línea de código sigue presente, comentada,
en la función extract_beta_from_R0() que puedes encontrar tanto en model_age_sir_vac.py
Como en model_age_sir.py. Para obtener las figuras que hasta ahora aparecen en el draft
bastaría con descomentarla. La nueva línea calcula beta como beta=(R0/gamma)*largest_eigenvalue(chi*contact). Lo que sale ahora es un beta algo más elevado que en el caso anterior y en consecuencia parece provocar un primer brote mucho más acentuado que en el caso anterior, y un segundo brote de tamaño inferior. Esto, parece ser, ha 
acabado afectando a la calidad de las correlaciones presentadas para el segundo brote.

