all:
	g++ main.cpp funcs.cpp adv.cpp out_vtk.cpp -o rbm2d.exe

cuda:
	nvcc -DUSEGPU main.cpp funcs.cpp adv.cpp out_vtk.cpp matrix.cu -o rbm2d.exe

clean:
	rm *.exe *.vtk 
