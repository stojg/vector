package vector

import (
	"fmt"
	"math"
)

var RealEpsilon float64

func init() {
	RealEpsilon = 0.00001
}

type Vector3 [3]float64

func (v *Vector3) String() string {
	return fmt.Sprintf("[%0.5f, %0.5f, %0.5f]", v[0], v[1], v[2])
}

var (
	UnitX = Vector3{1, 0, 0}
	UnitY = Vector3{0, 1, 0}
	UnitZ = Vector3{0, 0, 1}
)

func NewVector3(x, y, z float64) *Vector3 {
	e := &Vector3{}
	e[0] = x
	e[1] = y
	e[2] = z
	return e
}

func VectorZ() *Vector3 {
	return &Vector3{0, 0, 1}
}

func VectorY() *Vector3 {
	return &Vector3{0, 1, 0}
}

func VectorX() *Vector3 {
	return &Vector3{1, 0, 0}
}

func (v *Vector3) Clone() *Vector3 {
	return &Vector3{
		v[0],
		v[1],
		v[2],
	}
}

func (a *Vector3) Set(x, y, z float64) {
	a[0] = x
	a[1] = y
	a[2] = z
}

func (a *Vector3) Copy(b *Vector3) {
	a[0] = b[0]
	a[1] = b[1]
	a[2] = b[2]
}

func (v *Vector3) Clear() *Vector3 {
	v[0] = 0
	v[1] = 0
	v[2] = 0
	return v
}

func (a *Vector3) Add(b *Vector3) *Vector3 {
	a[0] += b[0]
	a[1] += b[1]
	a[2] += b[2]
	return a
}

func (a *Vector3) NewAdd(b *Vector3) *Vector3 {
	return &Vector3{
		a[0] + b[0],
		a[1] + b[1],
		a[2] + b[2],
	}
}

func (a *Vector3) Sub(b *Vector3) *Vector3 {
	a[0] -= b[0]
	a[1] -= b[1]
	a[2] -= b[2]
	return a
}

func (a *Vector3) NewSub(b *Vector3) *Vector3 {
	return &Vector3{
		a[0] - b[0],
		a[1] - b[1],
		a[2] - b[2],
	}
}

func (a *Vector3) AddScaledVector(b *Vector3, t float64) *Vector3 {
	if math.IsNaN(t) {
		panic("scale value passed to Vector3.AddScaledVector() is NaN")
	}
	a[0] += b[0] * t
	a[1] += b[1] * t
	a[2] += b[2] * t
	return a
}

func (a *Vector3) Inverse() *Vector3 {
	a[0] = -a[0]
	a[1] = -a[1]
	a[2] = -a[2]
	return a
}

func (a *Vector3) NewInverse() *Vector3 {
	return &Vector3{
		-a[0],
		-a[1],
		-a[2],
	}
}

func (a *Vector3) Length() float64 {
	return math.Sqrt(a[0]*a[0] + a[1]*a[1] + a[2]*a[2])
}

func (a *Vector3) SquareLength() float64 {
	return a[0]*a[0] + a[1]*a[1] + a[2]*a[2]
}

func (a *Vector3) Normalize() *Vector3 {
	length := a.Length()
	if length > 0 {
		a.Scale(1 / length)
	}
	return a
}

func (a *Vector3) Scale(alpha float64) *Vector3 {
	a[0] *= alpha
	a[1] *= alpha
	a[2] *= alpha
	return a
}

func (a *Vector3) NewScale(alpha float64) *Vector3 {
	return &Vector3{
		a[0] * alpha,
		a[1] * alpha,
		a[2] * alpha,
	}
}

func (a *Vector3) Dot(b *Vector3) float64 {
	return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]
}

func (a *Vector3) NewCross(vector *Vector3) *Vector3 {
	return &Vector3{
		a[1]*vector[2] - a[2]*vector[1],
		a[2]*vector[0] - a[0]*vector[2],
		a[0]*vector[1] - a[1]*vector[0],
	}

}

// VectorProduct aka cross product
func (a *Vector3) NewVectorProduct(vector *Vector3) *Vector3 {
	return a.NewCross(vector)
}

// ScalarProduct calculates and returns the scalar product of this vector
// with the given vector.
func (a *Vector3) ScalarProduct(vector *Vector3) float64 {
	return a[0]*vector[0] + a[1]*vector[1] + a[2]*vector[2]
}

func (a *Vector3) HadamardProduct(vector *Vector3) *Vector3 {
	a[0] *= vector[0]
	a[1] *= vector[1]
	a[2] *= vector[2]
	return a
}

func (a *Vector3) NewHadamardProduct(vector *Vector3) *Vector3 {
	return &Vector3{
		a[0] * vector[0],
		a[1] * vector[1],
		a[2] * vector[2],
	}
}

func (a *Vector3) Equals(z *Vector3) bool {
	diff := math.Abs(a[0] - z[0])
	if diff > RealEpsilon {
		return false
	}
	diff = math.Abs(a[1] - z[1])
	if diff > RealEpsilon {
		return false
	}
	diff = math.Abs(a[2] - z[2])
	if diff > RealEpsilon {
		return false
	}
	return true
}
