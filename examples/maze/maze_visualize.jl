include("maze.jl")

using ModernGL, GeometryTypes, GLAbstraction, GLWindow, Images, FileIO, Reactive

function rectangle(start, width, height, pos, cols, elms)
	indx = length(pos)
	
	push!(pos, start)
	push!(pos, (start[1] + width, start[2]))
	push!(pos, (start[1], start[2] - height))
	push!(pos, (start[1] + width, start[2] - height))

	push!(elms, (indx, indx+1, indx+2))
	push!(elms, (indx+1, indx+2, indx+3))
	
	for i=1:4; push!(cols, (0.0, 0.0, 0.0)); end
end

function agent_model(start, width, height, pos, cols, elms)
	indx = length(pos)
	push!(pos, (start[1] + width*0.5, start[2] - height*0.25))
	push!(pos, (start[1] + width*0.75, start[2] - height*0.75))
	push!(pos, (start[1] + width*0.25, start[2] - height*0.75))

	push!(cols, (1.0, 0.0, 0.0))
	push!(cols, (0.0, 0.0, 1.0))
	push!(cols, (0.0, 0.0, 1.0))

	push!(elms, (indx, indx+1, indx+2))
end

function paint(start, width, height, pos, cols, elms, clr)
	indx = length(pos)
	push!(pos, start)
	push!(pos, (start[1] + width, start[2]))
	push!(pos, (start[1], start[2] - height))
	push!(pos, (start[1] + width, start[2] - height))
	
	push!(elms, (indx, indx+1, indx+2))
	push!(elms, (indx+1, indx+2, indx+3))
	
	for i=1:4; push!(cols, clr); end

end

function key_callback(window, key, scancode, action, mode)
	if key == GLFW.KEY_ESCAPE && action == GLFW.PRESS
		GLFW.SetWindowShouldClose(window, true)
	else
		if key == GLFW.KEY_L && action == GLFW.PRESS
			left()
		elseif key == GLFW.KEY_R && action == GLFW.PRESS
			right()
		elseif key == GLFW.KEY_M && action == GLFW.PRESS
			move()
		elseif key == GLFW.KEY_S && action == GLFW.PRESS
			reset()
		end
	end
end	

function reset()
	global orientation
	global ro
	global a_ro
	global signal
	global window
	global a_model
	global env
	hm, wm,_ = size(env.maze)
	w = 2.0 / wm
	h = 2.0 / hm

	a_ro[:view] = translationmatrix(Vec{3, Float32}((-(wm / 2 - 0.5)*w, (hm/2 + 0.5 - env.start[1])*h , 0.0f0))) * scalematrix(Vec{3, Float32}((1.0/wm, 1.0/hm, 1.0f0)))

	push!(signal, rotationmatrix_z(deg2rad((orientation-1)*90)))
	orientation = 1

	Reactive.run_till_now()
	glClear(GL_COLOR_BUFFER_BIT)
	render(ro)
	render(a_ro)
	GLFW.SwapBuffers(window)
	sleep(0.01)
end

function rot(pos=1.0)
	global orientation
	global ro
	global a_ro
	global signal
	global window

	if pos == 1.0
		orientation = orientation == 1 ? 4 : orientation - 1
	else
		orientation = orientation == 4 ? 1 : orientation + 1
	end

	for i=1:30
		push!(signal, rotationmatrix_z(deg2rad(3*pos)))
		Reactive.run_till_now()
		glClear(GL_COLOR_BUFFER_BIT)
		render(ro)
		render(a_ro)
		GLFW.SwapBuffers(window)
		sleep(0.001)
	end

end

left() = rot(1.0)
right() = rot(-1.0)

function move()
	global env
	global orientation
	global signal
	global ro
	global a_ro
	global window
	hm, wm,_ = size(env.maze)
	w = 2.0 / wm
	h = 2.0 / hm
	wdx = orientation == 2 ? 1.0 : orientation == 4 ? -1.0 : 0.0
	hdx = orientation == 1 ? 1.0 : orientation == 3 ? -1.0 : 0.0
	for i=1:30
		a_ro[:view] = translationmatrix(Vec{3, Float32}((w/30.0 * wdx, h/30.0 * hdx,0.0f0))) * a_ro[:view]
		glClear(GL_COLOR_BUFFER_BIT)
		render(ro)
		render(a_ro)
		GLFW.SwapBuffers(window)
		sleep(0.001)
	end
end

function main()

	startpng = load("start.png")
	checkeredpng = load("checkered.png")

	wm = 10
	hm = 10
	global env = MazeEnv((hm,wm))
	agent = QLearner(env)

	global orientation = 1

	global window = create_glcontext("Maze Solving", resolution=(800, 600))

	vao = glGenVertexArrays()
	glBindVertexArray(vao)

	w = 2.0 / wm
	h = 2.0 / hm

	positions = Point{2, Float32}[]
	clrs = Vec3f0[]
	elements = Face{3, UInt32, -1}[]

	a_positions = Point{2, Float32}[]
	a_clrs = Vec3f0[]
	a_elements = Face{3, UInt32, -1}[]
	
	paint((-1, 1 - (env.start[1]-1)*h), w, h, positions, clrs, elements, (1.0, 1.0, 0.0))
	paint((-1 + (wm - 1)*w, 1 - (env.goal[1]-1)*h), w, h, positions, clrs, elements, (0.0, 1.0, 0.0))
	
	agent_model((-1, 1), 2.0, 2.0, a_positions, a_clrs, a_elements)
	
	for i=1:wm
		for j=1:hm
			start = (-1 + (i-1)*w, 1 - (j-1)*h)
			if env.maze[j, i, 1] == 0
				rectangle(start, w, h / 10.0, positions, clrs, elements)
			end
			if env.maze[j, i, 2] == 0 && i == wm
				s = (start[1] + w * 0.9, start[2])
				rectangle(s, w / 10.0, h, positions, clrs, elements)
			end
			if env.maze[j, i, 3] == 0 && j == hm
				s = (start[1], start[2] - h * 0.9)
				rectangle(s, w, h / 10.0, positions, clrs, elements)
			end
			if env.maze[j, i, 4] == 0
				rectangle(start, w / 10.0, h, positions, clrs, elements)
			end
		end
	end


	vertex_source= vert"""
	# version 150
	in vec2 position;
	in vec3 color;

	uniform mat4 model;
	uniform mat4 view;
	uniform mat4 proj;

	out vec3 Color;
	void main()
	{
	Color = color;
	gl_Position = proj * view * model * vec4(position,0.0, 1.0);
	}
	"""
	fragment_source = frag"""
	# version 150
	in vec3 Color;
	out vec4 outColor;
	void main()
	{
	outColor = vec4(Color, 1.0);
	}
	"""
	
	global signal = Signal(rotate(0f0, Vec((0,0,1f0))))
	model = rotate(0f0, Vec((0,0,1f0)))
	view = rotate(0f0, Vec((0,0,1f0)))
	global a_model = foldp(*, rotate(0f0, Vec((0,0,1f0))), signal)
	a_view = translationmatrix(Vec((-(wm / 2 - 0.5)*w, (hm/2 + 0.5 - env.start[1])*h , 0.0f0))) * scalematrix(Vec((1.0/wm, 1.0/hm, 1.0f0)))

	proj = orthographicprojection(Float32, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0)
	

	bufferdict = Dict(:position=>GLBuffer(positions),
	:color=>GLBuffer(clrs),
	:indexes=>indexbuffer(elements),
	:model=>model,
	:view=>view,
	:proj=>proj,
	)		

	a_bufferdict = Dict(:position=>GLBuffer(a_positions),
	:color=>GLBuffer(a_clrs),
	:indexes=>indexbuffer(a_elements),
	:model=>a_model,
	:view=>a_view,
	:proj=>proj
	)		


	global ro = std_renderobject(bufferdict, LazyShader(vertex_source, fragment_source))
	global a_ro = std_renderobject(a_bufferdict, LazyShader(vertex_source, fragment_source))
	
	GLFW.SetKeyCallback(window, key_callback)

	glClearColor(1.0,1.0,1.0,1.0)

	while !GLFW.WindowShouldClose(window)
		glClear(GL_COLOR_BUFFER_BIT)
		render(ro)
		render(a_ro)
		GLFW.SwapBuffers(window)
		GLFW.PollEvents()
	end			
end

main()
