 import dream

dream.set\_electric\_field(0.6)
dream.set\_background\_density(5e19)
dream.set\_temperature(1e3)
dream.set\_ion\_species("D", 1)
dream.disable\_hot\_tail\_grid()
dream.set\_collision\_frequency\_mode("ultra-relativistic")
dream.include\_runaway\_electrons()
dream.set\_runaway\_ radial\_points(50)
dream.set\_runaway\_momentum\_points(100)
dream.set\_runaway\_maximum\_momentum(0.5)
dream.set\_runaway\_advection\_interpolation("flux\_limiters")
dream.set\_runaway\_initialization("isotropic")
dream.set\_magnetic\_field(5)
dream.set\_radial\_grid(0.22, 0.22, 1)
dream.set\_solver("nonlinear", verbose=True, reltol=1e-4)
dream.include\_fluid\_effects()
dream.set\_time\_stepper(1e-1, 20)
dream.save\_settings("dream\_settings.h5")