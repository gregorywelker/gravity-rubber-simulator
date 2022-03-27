// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : Welker Gergo
// Neptun : ECKCA4
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================

#include "framework.h"

class PhongShader : public GPUProgram
{

	const char *vertexSource = R"(
		#version 330
		precision highp float;

		uniform mat4  MVP, M, Minv;
		uniform vec3  wLiDir, pl1pos, pl2pos;
		uniform vec3  wEye;

		layout(location = 0) in vec3  vtxPos;
		layout(location = 1) in vec3  vtxNorm;
		layout(location = 2) in vec2  vtxUV;

		out vec3 wNormal;
		out vec3 wView;
		out vec3 wLight;
		out float pl1dist, pl2dist;
		out float yPos;

		void main() {
		   gl_Position = vec4(vtxPos, 1) * MVP;
		   vec4 wPos = vec4(vtxPos, 1) * M;
		   wLight  = wLiDir;
		   wView   = wEye - wPos.xyz;
		   wNormal = (Minv * vec4(vtxNorm, 0)).xyz;
		   pl1dist = length(vtxPos - pl1pos);
		   pl2dist = length(vtxPos - pl2pos);
		   yPos = vtxPos.y;
		}
	)";

	const char *fragmentSource = R"(
		#version 330
		precision highp float;

		uniform vec3 kd, ks, ka;
		uniform vec3 lAmbient, lDirectional, pl1color, pl2color;
		uniform float shine, pl1intensity, pl2intensity;

		in  vec3 wNormal;
		in  vec3 wView;
		in  vec3 wLight;
		in float pl1dist, pl2dist, yPos;

		in vec2 texcoord;
		out vec4 fragmentColor;

		void main() {
			vec3 N = normalize(wNormal);
			vec3 V = normalize(wView);
			vec3 L = normalize(wLight);
			vec3 H = normalize(L + V);
			float cost = max(dot(N,L), 0), cosd = max(dot(N,H), 0);
			vec3 color = ka * lAmbient + (kd * cost + ks * pow(cosd,shine)) * lDirectional + pl1color * pl1intensity / (pl1dist * pl1dist) +  pl2color * pl2intensity / (pl2dist * pl2dist);
			
			int divs = 4;
			float firstDiv = -0.25f;
			float divSize = firstDiv - 0.1f;
			int i = 1;
			for(; i < divs; i++)
			{
				if(yPos > divSize * i && yPos < firstDiv)
				{
					color *= 1.0f - 1.0f / ((divs - i) + 1);
					break;
				}
			}
			if(i >= divs && yPos < firstDiv)
			{
				color *= 0.1f;
			}
			fragmentColor = vec4(color, 1);
		}
	)";

public:
	PhongShader() { create(vertexSource, fragmentSource, "fragmentColor"); }
};

PhongShader *gpuProgram;

vec4 RodriguesFormula(vec4 v1, vec4 v2)
{
	vec3 d1(v1.y, v1.z, v1.w);
	vec3 d2(v2.y, v2.z, v2.w);
	vec3 prod = v1.x * d2 + v2.x * d1 + cross(d1, d2);
	return vec4(v1.x * v2.x - dot(d1, d2), prod.x, prod.y, prod.z);
}

vec3 CustomQuaternionRotation(vec3 point, float t)
{
	vec3 axis = vec3(cos(t) / 2.0f, sin(t) / 2.0f, sqrt(3.0f / 4.0f)) * sin(t / 4.0f);
	vec4 q(cosf(t / 4.0f), axis.x, axis.y, axis.z);
	vec4 rotPoint(0.0f, point.x, point.y, point.z);
	vec4 qinv = vec4(cosf(t / 4.0f), (-1.0f) * axis.x, (-1.0f) * axis.y, (-1.0f) * axis.z) / dot(q, q);

	vec4 prod1 = RodriguesFormula(q, rotPoint);
	vec4 prod2 = RodriguesFormula(prod1, qinv);

	return vec3(prod2.y, prod2.z, prod2.w);
}

vec3 operator/(vec3 num, vec3 denom)
{
	return vec3(num.x / denom.x, num.y / denom.y, num.z / denom.z);
}

struct Camera
{
	vec3 wEye, wLookat, wVup;
	float fov, asp, fp, bp;

public:
	Camera()
	{
		asp = 1;
		fov = 90.0f * (float)M_PI / 180.0f;
		fp = 0.1;
		bp = 100;
	}
	mat4 V()
	{
		vec3 w = normalize(wEye - wLookat);
		vec3 u = normalize(cross(wVup, w));
		vec3 v = cross(w, u);
		return TranslateMatrix(-wEye) * mat4(u.x, v.x, w.x, 0,
											 u.y, v.y, w.y, 0,
											 u.z, v.z, w.z, 0,
											 0, 0, 0, 1);
	}
	mat4 P()
	{
		return mat4(1 / (tan(fov / 2) * asp), 0, 0, 0,
					0, 1 / tan(fov / 2), 0, 0,
					0, 0, -(fp + bp) / (bp - fp), -1,
					0, 0, -2 * fp * bp / (bp - fp), 0);
	}
	void SetUniform()
	{
		int location = glGetUniformLocation(gpuProgram->getId(), "wEye");
		if (location >= 0)
			glUniform3fv(location, 1, &wEye.x);
		else
			printf("uniform wEye cannot be set\n");
	}
};

Camera camera;

struct Material
{
	vec3 kd, ks, ka;
	float shininess;

	void SetUniform()
	{
		gpuProgram->setUniform(kd, "kd");
		gpuProgram->setUniform(ks, "ks");
		gpuProgram->setUniform(ka, "ka");
		int location = glGetUniformLocation(gpuProgram->getId(), "shine");
		if (location >= 0)
			glUniform1f(location, shininess);
		else
			printf("uniform shininess cannot be set\n");
	}
};

struct PointLight
{
	vec3 color, position;
	float intensity;
	std::string positionTarget, colorTarget, intensityTarget;
	PointLight(vec3 color, vec3 position, float intensity,
			   std::string colorTarget, std::string positionTarget, std::string intensityTarget) : color(color), position(position), intensity(intensity),
																								   positionTarget(positionTarget), colorTarget(colorTarget),
																								   intensityTarget(intensityTarget){};

	void SetUniform()
	{
		gpuProgram->setUniform(position, positionTarget);
		gpuProgram->setUniform(color, colorTarget);
		gpuProgram->setUniform(intensity, intensityTarget);
	}
};

struct WorldDirectionalLight
{
	vec3 ambientLight, directionalLight, wLightDir;

	WorldDirectionalLight(vec3 dir) : ambientLight(0.5f, 0.9f, 0.75f), directionalLight(1.9f, 1.8f, 2.9f), wLightDir(dir) {}

	void SetUniform()
	{
		gpuProgram->setUniform(ambientLight, "lAmbient");
		gpuProgram->setUniform(directionalLight, "lDirectional");
		gpuProgram->setUniform(wLightDir, "wLiDir");
	}
};

struct VertexData
{
	vec3 position, normal;
	vec2 texcoord;
};

class Geometry
{
	unsigned int vao, type;

protected:
	int nVertices;

public:
	Geometry(unsigned int _type)
	{
		type = _type;
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);
	}
	void Draw(mat4 M, mat4 Minv)
	{
		mat4 MVP = M * camera.V() * camera.P();
		gpuProgram->setUniform(MVP, "MVP");
		gpuProgram->setUniform(M, "M");
		gpuProgram->setUniform(Minv, "Minv");
		glBindVertexArray(vao);
		glDrawArrays(type, 0, nVertices);
	}
};

class ParamSurface : public Geometry
{
public:
	ParamSurface() : Geometry(GL_TRIANGLES) {}

	virtual VertexData GenVertexData(float u, float v) = 0;

	void Create(int N = 16, int M = 16)
	{
		unsigned int vbo;
		glGenBuffers(1, &vbo);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		nVertices = N * M * 6;
		std::vector<VertexData> vtxData;
		for (int i = 0; i < N; i++)
		{
			for (int j = 0; j < M; j++)
			{
				vtxData.push_back(GenVertexData((float)i / N, (float)j / M));
				vtxData.push_back(GenVertexData((float)(i + 1) / N, (float)j / M));
				vtxData.push_back(GenVertexData((float)i / N, (float)(j + 1) / M));
				vtxData.push_back(GenVertexData((float)(i + 1) / N, (float)j / M));
				vtxData.push_back(GenVertexData((float)(i + 1) / N, (float)(j + 1) / M));
				vtxData.push_back(GenVertexData((float)i / N, (float)(j + 1) / M));
			}
		}
		glBufferData(GL_ARRAY_BUFFER, nVertices * sizeof(VertexData), &vtxData[0], GL_STATIC_DRAW);

		glEnableVertexAttribArray(0);
		glEnableVertexAttribArray(1);
		glEnableVertexAttribArray(2);

		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void *)offsetof(VertexData, position));
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void *)offsetof(VertexData, normal));
		glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void *)offsetof(VertexData, texcoord));
	}
};

struct Weight
{
	vec2 pos;
	float w;

	Weight(vec2 pos, float w)
	{
		this->pos = pos;
		this->w = w;
	};
};
std::vector<Weight *> weights;
float currentWeight = 0.2f;

struct IDisplayable
{
	virtual void Display() = 0;
};

struct IPhysicsObject
{
	vec3 position = vec3(0, 0, 0), velocity = vec3(0, 0, 0);
	bool isActive = false;
	bool simulationEnded = false;
	float maxVelocity;
	virtual void Simulate(float dt) = 0;
};

class RubberSheet : public ParamSurface, public IDisplayable
{
	float size;
	Material *material;

public:
	RubberSheet();
	vec3 GetPositionOnPlane(float u, float v);
	vec3 GetNormal(float u, float v);
	float GetYGravityDistortion(vec3 pos);
	VertexData GenVertexData(float u, float v);
	void Display();
	void AddWeight(vec2 w);
	vec2 GetMouseProjectionOnSheet(vec2 w);
	float GetSize();
};

RubberSheet::RubberSheet()
{
	size = 10;
	Create(80, 80);

	material = new Material();
	material->kd = vec3(0.3f, 0.2f, 0.1f);
	material->ks = vec3(0.12f, 0.07f, 0.02f);
	material->ka = vec3(0.05f, 0.02f, 0.01f);
	material->shininess = 70;
}

vec3 RubberSheet::GetPositionOnPlane(float u, float v)
{
	return vec3((u - 0.5) * 2, 0, (v - 0.5) * 2) * size;
}

vec3 RubberSheet::GetNormal(float u, float v)
{
	vec3 pOnPlane = GetPositionOnPlane(u, v);
	vec3 p = vec3(pOnPlane.x, GetYGravityDistortion(pOnPlane), pOnPlane.z);

	vec3 tar1OnPlane = GetPositionOnPlane(u + 0.001f, v);
	vec3 tar1 = vec3(tar1OnPlane.x, GetYGravityDistortion(tar1OnPlane), tar1OnPlane.z);

	vec3 tar2OnPlane = GetPositionOnPlane(u, v - 0.001f);
	vec3 tar2 = vec3(tar2OnPlane.x, GetYGravityDistortion(tar2OnPlane), tar2OnPlane.z);

	vec3 dir1 = tar1 - p;
	vec3 dir2 = tar2 - p;
	vec3 nrm = normalize(cross(dir1, dir2));
	return nrm;
}

float RubberSheet::GetYGravityDistortion(vec3 pos)
{
	if (weights.size() <= 0)
	{
		return 0;
	}
	else
	{
		vec2 projection = vec2(pos.x, pos.z);
		float y = 0;
		for (size_t i = 0; i < weights.size(); i++)
		{
			y += weights[i]->w * (-1) / (length(weights[i]->pos - projection) * length(weights[i]->pos - projection) + size * 0.005f);
		}
		return y;
	}
}

VertexData RubberSheet::GenVertexData(float u, float v)
{
	VertexData vd;
	vec3 posOnPlane = GetPositionOnPlane(u, v);
	vd.position = vec3(posOnPlane.x, GetYGravityDistortion(posOnPlane), posOnPlane.z);
	vd.normal = GetNormal(u, v);
	vd.texcoord = vec2(u, v);
	return vd;
}

void RubberSheet::Display()
{
	mat4 unit = TranslateMatrix(vec3(0, 0, 0));
	material->SetUniform();
	this->Draw(unit, unit);
}

void RubberSheet::AddWeight(vec2 w)
{
	weights.push_back(new Weight(GetMouseProjectionOnSheet(w), currentWeight));
	currentWeight += 0.075f;
}

vec2 RubberSheet::GetMouseProjectionOnSheet(vec2 w)
{
	return vec2(((w.x) * size * -1), ((w.y) * size));
}

float RubberSheet::GetSize()
{
	return size;
}

class Ball : public ParamSurface, public IPhysicsObject, public IDisplayable
{
	float r;
	Material *material;
	RubberSheet *terrain;

public:
	Ball(float _r)
	{
		material = new Material();
		material->kd = vec3(static_cast<float>(rand()) / static_cast<float>(RAND_MAX),
							static_cast<float>(rand()) / static_cast<float>(RAND_MAX),
							static_cast<float>(rand()) / static_cast<float>(RAND_MAX));
		material->ks = vec3(1.3f, 1.3f, 2.3f);
		material->ka = vec3(0.1f, 0.1f, 0.1f);
		material->shininess = 100;

		r = _r;
		position = vec3(9.4f, _r, -9.4f);
		Create(20, 20);
	}

	VertexData GenVertexData(float u, float v)
	{
		VertexData vd;
		vd.normal = vec3(cosf(u * 2.0f * M_PI) * sin(v * M_PI), sinf(u * 2.0f * M_PI) * sinf(v * M_PI), cosf(v * M_PI));
		vd.position = vd.normal * r;
		vd.texcoord = vec2(u, v);
		return vd;
	}
	void Simulate(float dt)
	{
		if (isActive)
		{

			vec2 planeProjection = vec2((position.x / terrain->GetSize()) / 2 + 0.5f, (position.z / terrain->GetSize()) / 2 + 0.5f);

			vec3 nrm = terrain->GetNormal(planeProjection.x, planeProjection.y);

			float gravityY = -9.81f * dt;
			velocity = velocity + vec3(nrm.x, 0, nrm.z) * dt;

			if (length(velocity) > maxVelocity)
			{
				velocity = velocity / (1 / maxVelocity);
			}

			position = position + velocity + vec3(0, gravityY, 0);

			if ((position.y - r) < terrain->GetYGravityDistortion(position))
			{
				position = vec3(position.x, terrain->GetYGravityDistortion(position) + r, position.z);
			}

			if (position.x > terrain->GetSize())
			{
				position = vec3(position.x - terrain->GetSize() * 2, position.y, position.z);
			}
			if (position.x < -terrain->GetSize())
			{
				position = vec3(position.x + terrain->GetSize() * 2, position.y, position.z);
			}
			if (position.z > terrain->GetSize())
			{
				position = vec3(position.x, position.y, position.z - terrain->GetSize() * 2);
			}
			if (position.z < -terrain->GetSize())
			{
				position = vec3(position.x, position.y, position.z + terrain->GetSize() * 2);
			}

			if (position.y < -2)
			{
				simulationEnded = true;
				isActive = false;
			}
		}
	}
	void Display()
	{
		if (!simulationEnded)
		{
			mat4 M = TranslateMatrix(position);
			mat4 Minv = TranslateMatrix(vec3(1, 1, 1) / position);
			material->SetUniform();
			this->Draw(M, Minv);
		}
	}
	void SetTerrain(RubberSheet *t)
	{
		this->terrain = t;
	}
	void SetInitialVelocity(vec2 mousePos)
	{
		vec2 proj = terrain->GetMouseProjectionOnSheet(mousePos);
		this->velocity = 0.65f * normalize(vec3(proj.x, 0, proj.y) - vec3(position.x, 0, position.z));
		maxVelocity = length(velocity);
	}
	vec3 GetPosition()
	{
		return position;
	}
};

class Scene
{
	RubberSheet *terrain;
	std::vector<IPhysicsObject *> physicsObjects;
	std::vector<IDisplayable *> sceneObjects;

	WorldDirectionalLight *wDirLight;

	Ball *activeBall;

	Ball *target;
	bool cameraFollow;

	PointLight *pl1;
	PointLight *pl2;

public:
	void Build()
	{
		CameraTopViewSetup();

		wDirLight = new WorldDirectionalLight(vec3(20, 35, 5));

		Ball *b = new Ball(0.25f);
		physicsObjects.push_back(b);
		sceneObjects.push_back(b);
		activeBall = b;
		target = activeBall;

		terrain = new RubberSheet();

		pl1 = new PointLight(vec3(0.25f, 0.1f, 0.4f), vec3(5.2f, 2, 4.4f), 10, std::string("pl1color"), std::string("pl1pos"), std::string("pl1intensity"));
		pl2 = new PointLight(vec3(0.45f, 0.2f, 0.05f), vec3(-1.25f, 3, -3.75f), 10, std::string("pl2color"), std::string("pl2pos"), std::string("pl2intensity"));
	}
	void CameraTopViewSetup()
	{
		camera.wEye = vec3(0, 10, 0);
		camera.wLookat = vec3(0, 0, 0);
		camera.wVup = vec3(0, 0, 1);
	}
	void AddBall(vec2 mousePos)
	{

		activeBall->SetTerrain(terrain);
		activeBall->SetInitialVelocity(mousePos);
		activeBall->isActive = true;

		target = activeBall;

		Ball *b = new Ball(0.25f);
		physicsObjects.push_back(b);
		sceneObjects.push_back(b);
		activeBall = b;
	}

	void SimulateAll(float t)
	{
		static float tprev = 0;
		float dt = t - tprev;
		tprev = t;
		for (size_t i = 0; i < physicsObjects.size(); i++)
		{
			physicsObjects[i]->Simulate(dt);
		}

		pl1->position = CustomQuaternionRotation(pl1->position, dt / 20.0f);
		pl2->position = CustomQuaternionRotation(pl2->position, dt / 20.0f);
	}
	void Render()
	{
		if (cameraFollow && !target->simulationEnded)
		{
			camera.wEye = target->position - (target->isActive ? target->velocity : vec3(-0.4f, 0, 0.4f)) * 3 + vec3(0, 0.65f, 0);
			camera.wLookat = target->position;
			camera.wVup = vec3(0, 1, 0);
		}
		else
		{
			cameraFollow = false;
			CameraTopViewSetup();
		}

		camera.SetUniform();
		pl1->SetUniform();
		pl2->SetUniform();
		wDirLight->SetUniform();

		terrain->Display();
		for (size_t i = 0; i < sceneObjects.size(); i++)
		{
			sceneObjects[i]->Display();
		}
	}
	RubberSheet *GetTerrain()
	{
		return terrain;
	}

	void RetesellateTerrain()
	{
		terrain = new RubberSheet();
	}

	void CameraFollowSetup()
	{
		if (target->simulationEnded)
		{
			target = activeBall;
		}
		cameraFollow = true;
	}
};

Scene scene;

void onInitialization()
{
	srand(0);
	glViewport(0, 0, windowWidth, windowHeight);
	glEnable(GL_DEPTH_TEST);
	glDisable(GL_CULL_FACE);

	gpuProgram = new PhongShader();

	scene.Build();
}

void onDisplay()
{
	glClearColor(0.3f, 0.475f, 0.65f, 0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	scene.Render();
	glutSwapBuffers();
}

void onKeyboard(unsigned char key, int pX, int pY)
{
	if (key == 'd')
		glutPostRedisplay();

	if (key == 32)
	{
		scene.CameraFollowSetup();
	}
}

void onKeyboardUp(unsigned char key, int pX, int pY)
{
}

void onMouseMotion(int pX, int pY)
{
	float cX = 2.0f * pX / windowWidth - 1;
	float cY = 1.0f - 2.0f * pY / windowHeight;
	printf("Mouse moved to (%3.2f, %3.2f)\n", cX, cY);
}

void onMouse(int button, int state, int pX, int pY)
{
	float cX = 2.0f * pX / windowWidth - 1;
	float cY = 1.0f - 2.0f * pY / windowHeight;

	char *buttonStat;
	switch (state)
	{
	case GLUT_DOWN:
		buttonStat = "pressed";
		break;
	case GLUT_UP:
		buttonStat = "released";
		switch (button)
		{
		case GLUT_LEFT_BUTTON:
			printf("Left button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY);
			scene.AddBall(vec2(cX, cY));
			break;
		case GLUT_MIDDLE_BUTTON:
			printf("Middle button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY);
			break;
		case GLUT_RIGHT_BUTTON:
			printf("Right button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY);
			scene.GetTerrain()->AddWeight(vec2(cX, cY));
			scene.RetesellateTerrain();
			break;
		}

		break;
	}
}

void onIdle()
{
	long time = glutGet(GLUT_ELAPSED_TIME);
	float sec = time / 30.0f;
	scene.SimulateAll(sec);
	glutPostRedisplay();
}