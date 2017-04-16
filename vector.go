package vector

import (
	"fmt"
	"math"
)

// RealEpsilon gives an upper bound on the relative error due to rounding in floating point arithmetic.
var RealEpsilon float64

func init() {
	RealEpsilon = 0.00001
}

// Vector3 is a 3 dimensional vector
type Vector3 [3]float64

// String return a string representation of a vector with 5 decimals per axis
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

func Zero() *Vector3 {
	return &Vector3{0, 0, 0}
}

func Z() *Vector3 {
	return &Vector3{0, 0, 1}
}

func Y() *Vector3 {
	return &Vector3{0, 1, 0}
}

func X() *Vector3 {
	return &Vector3{1, 0, 0}
}

func (v *Vector3) Clone() *Vector3 {
	return &Vector3{
		v[0],
		v[1],
		v[2],
	}
}

func (v *Vector3) Set(x, y, z float64) {
	v[0] = x
	v[1] = y
	v[2] = z
}

func (v *Vector3) Copy(b *Vector3) {
	v[0] = b[0]
	v[1] = b[1]
	v[2] = b[2]
}

func (v *Vector3) Clear() *Vector3 {
	v[0] = 0
	v[1] = 0
	v[2] = 0
	return v
}

func (v *Vector3) Add(b *Vector3) *Vector3 {
	v[0] += b[0]
	v[1] += b[1]
	v[2] += b[2]
	return v
}

func (v *Vector3) NewAdd(b *Vector3) *Vector3 {
	return &Vector3{
		v[0] + b[0],
		v[1] + b[1],
		v[2] + b[2],
	}
}

func (v *Vector3) Sub(b *Vector3) *Vector3 {
	v[0] -= b[0]
	v[1] -= b[1]
	v[2] -= b[2]
	return v
}

func (v *Vector3) NewSub(b *Vector3) *Vector3 {
	return &Vector3{
		v[0] - b[0],
		v[1] - b[1],
		v[2] - b[2],
	}
}

func (v *Vector3) AddScaledVector(b *Vector3, t float64) *Vector3 {
	if math.IsNaN(t) {
		panic("scale value passed to Vector3.AddScaledVector() is NaN")
	}
	v[0] += b[0] * t
	v[1] += b[1] * t
	v[2] += b[2] * t
	return v
}

func (v *Vector3) Inverse() *Vector3 {
	v[0] = -v[0]
	v[1] = -v[1]
	v[2] = -v[2]
	return v
}

func (v *Vector3) NewInverse() *Vector3 {
	return &Vector3{
		-v[0],
		-v[1],
		-v[2],
	}
}

func (v *Vector3) Length() float64 {
	return math.Sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])
}

func (v *Vector3) SquareLength() float64 {
	return v[0]*v[0] + v[1]*v[1] + v[2]*v[2]
}

func (v *Vector3) Normalize() *Vector3 {
	length := v.Length()
	if length > 0 {
		v.Scale(1 / length)
	}
	return v
}

func (v *Vector3) Scale(t float64) *Vector3 {
	v[0] *= t
	v[1] *= t
	v[2] *= t
	return v
}

func (v *Vector3) NewScale(t float64) *Vector3 {
	return &Vector3{
		v[0] * t,
		v[1] * t,
		v[2] * t,
	}
}

func (v *Vector3) Dot(b *Vector3) float64 {
	return v[0]*b[0] + v[1]*b[1] + v[2]*b[2]
}

// NewCross aka VectorProduct "%"
func (v *Vector3) NewCross(b *Vector3) *Vector3 {
	return &Vector3{
		v[1]*b[2] - v[2]*b[1],
		v[2]*b[0] - v[0]*b[2],
		v[0]*b[1] - v[1]*b[0],
	}

}

// NewVectorProduct aka cross product
func (v *Vector3) NewVectorProduct(b *Vector3) *Vector3 {
	return v.NewCross(b)
}

// ScalarProduct calculates and returns the scalar product of this vector
// with the given vector.
func (v *Vector3) ScalarProduct(b *Vector3) float64 {
	return v[0]*b[0] + v[1]*b[1] + v[2]*b[2]
}

func (v *Vector3) HadamardProduct(b *Vector3) *Vector3 {
	v[0] *= b[0]
	v[1] *= b[1]
	v[2] *= b[2]
	return v
}

func (v *Vector3) NewHadamardProduct(b *Vector3) *Vector3 {
	return &Vector3{
		v[0] * b[0],
		v[1] * b[1],
		v[2] * b[2],
	}
}

func (v *Vector3) Equals(b *Vector3) bool {
	diff := math.Abs(v[0] - b[0])
	if diff > RealEpsilon {
		return false
	}
	diff = math.Abs(v[1] - b[1])
	if diff > RealEpsilon {
		return false
	}
	diff = math.Abs(v[2] - b[2])
	return diff < RealEpsilon
}

// http://pastebin.com/fAFp6NnN
func (v *Vector3) Rotate(q *Quaternion) *Vector3 {
	num12 := q.I + q.I
	num2 := q.J + q.J
	num := q.K + q.K
	num11 := q.R * num12
	num10 := q.R * num2
	num9 := q.R * num
	num8 := q.I * num12
	num7 := q.I * num2
	num6 := q.I * num
	num5 := q.J * num2
	num4 := q.J * num
	num3 := q.K * num
	num15 := ((v[0] * ((1.0 - num5) - num3)) + (v[1] * (num7 - num9))) + (v[2] * (num6 + num10))
	num14 := ((v[0] * (num7 + num9)) + (v[1] * ((1.0 - num8) - num3))) + (v[2] * (num4 - num11))
	num13 := ((v[0] * (num6 - num10)) + (v[1] * (num4 + num11))) + (v[2] * ((1.0 - num8) - num5))

	v[0] = num15
	v[1] = num14
	v[2] = num13
	return v
}

func (v *Vector3) NewRotate(q *Quaternion) *Vector3 {
	return v.Clone().Rotate(q)
}

// Matrix3 is a 3x3 dimensional matrix
type Matrix3 [9]float64

// SetFromComponents sets the matrix from the given three vectors components. These are arranged as
// the three columns of the matrix
func (m *Matrix3) SetFromComponents(a, b, c *Vector3) {
	m[0] = a[0]
	m[1] = b[0]
	m[2] = c[0]
	m[3] = a[1]
	m[4] = b[1]
	m[5] = c[1]
	m[6] = a[2]
	m[7] = b[2]
	m[8] = c[2]
}

func (m *Matrix3) Transform(v *Vector3) *Vector3 {
	return &Vector3{
		v[0]*m[0] + v[1]*m[1] + v[2]*m[2],
		v[0]*m[3] + v[1]*m[4] + v[2]*m[5],
		v[0]*m[6] + v[1]*m[7] + v[2]*m[8],
	}
}

// TransformTranspose is a convenience method that combines the effect of transforming a vector by
// the transpose of a matrix.
// It works by performing a regular matrix transformation, but selecting the components of matrix in
// row rather than column order.
func (m *Matrix3) TransformTranspose(v *Vector3) *Vector3 {
	return &Vector3{
		v[0]*m[0] + v[1]*m[3] + v[2]*m[6],
		v[0]*m[1] + v[1]*m[4] + v[2]*m[7],
		v[0]*m[2] + v[1]*m[5] + v[2]*m[8],
	}
}

func (m *Matrix3) TransformMatrix3(b *Matrix3) *Matrix3 {
	newMatrix := &Matrix3{}
	newMatrix[0] = m[0]*b[0] + m[1]*b[3] + m[2] + b[6]
	newMatrix[1] = m[0]*b[1] + m[1]*b[4] + m[2] + b[7]
	newMatrix[2] = m[0]*b[2] + m[1]*b[5] + m[2] + b[8]
	newMatrix[3] = m[3]*b[0] + m[4]*b[3] + m[5] + b[6]
	newMatrix[4] = m[3]*b[1] + m[4]*b[5] + m[5] + b[7]
	newMatrix[5] = m[3]*b[2] + m[4]*b[6] + m[5] + b[8]
	newMatrix[6] = m[6]*b[0] + m[7]*b[3] + m[8] + b[6]
	newMatrix[7] = m[6]*b[1] + m[7]*b[4] + m[8] + b[7]
	newMatrix[8] = m[6]*b[2] + m[7]*b[5] + m[8] + b[8]
	return newMatrix
}

// Returns a new matrix containing the inverse of this matrix
func (m *Matrix3) SetInverse(b *Matrix3) {

	t1 := b[0] * b[4]
	t2 := b[0] * b[5]
	t3 := b[1] * b[3]
	t4 := b[2] * b[3]
	t5 := b[1] * b[6]
	t6 := b[2] * b[6]

	det := t1*b[8] - t2*b[7] - t3*b[8] + t4*b[7] + t5*b[5] - t6*b[4]

	// make sure the determinant is non zero
	if det == 0 {
		return
	}
	invd := 1 / det

	m[0] = (b[4]*b[8] - b[5]*b[7]) * invd
	m[1] = -(b[1]*b[8] - b[2]*b[7]) * invd
	m[2] = (b[1]*b[5] - b[2]*b[4]) * invd

	m[3] = -(b[3]*b[8] - b[5]*b[6]) * invd
	m[4] = (b[0]*b[8] - t6) * invd
	m[5] = -(t2 - t4) * invd

	m[6] = (b[3]*b[7] - b[4]*b[6]) * invd
	m[7] = -(b[0]*b[7] - t5) * invd
	m[8] = (t1 - t3) * invd

}

// Returns a new matrix containing the inverse of this matrix, If a matrix represents a rotation the
// inverse is it's transpose. This very handy when we want to transforming coordinates from
// world -> local space and inverse of the matrix would do local -> world space.
func (m *Matrix3) NewInverse() *Matrix3 {
	result := &Matrix3{}
	result.SetInverse(m)
	return result
}

func (m *Matrix3) Invert() {
	m.SetInverse(m)
}

/**
 * Sets the value of the matrix from inertia tensor values.
 */
func (m *Matrix3) setInertiaTensorCoeffs(ix, iy, iz, ixy, ixz, iyz float64) {
	m[0] = ix
	m[1] = -ixy
	m[3] = -ixy
	m[2] = -ixz
	m[6] = -ixz
	m[4] = iy
	m[5] = -iyz
	m[7] = -iyz
	m[8] = iz
}

/**
 * Sets the value of the matrix as an inertia tensor of
 * a rectangular block aligned with the body's coordinate
 * system with the given axis half-sizes and mass.
 */
func (m *Matrix3) SetBlockInertiaTensor(halfSizes *Vector3, mass float64) {
	squares := halfSizes.NewHadamardProduct(halfSizes)

	m.setInertiaTensorCoeffs(
		0.3*mass*(squares[1]+squares[2]),
		0.3*mass*(squares[0]+squares[2]),
		0.3*mass*(squares[0]+squares[1]),
		0,
		0,
		0,
	)
}

func (m *Matrix3) SetTranspose(b *Matrix3) {
	m[0] = b[0]
	m[1] = b[3]
	m[2] = b[6]
	m[3] = b[1]
	m[4] = b[4]
	m[5] = b[7]
	m[6] = b[2]
	m[7] = b[5]
	m[8] = b[8]
}

func (m *Matrix3) Transpose() *Matrix3 {
	result := &Matrix3{}
	result.SetTranspose(m)
	return result
}

func (m *Matrix3) SetOrientation(q *Quaternion) {
	m[0] = 1 - (2*q.J*q.J + 2*q.K*q.K)
	m[1] = 2*q.I*q.J + 2*q.K*q.R
	m[2] = 2*q.I*q.K - 2*q.J*q.R
	m[3] = 2*q.I*q.J - 2*q.K*q.R
	m[4] = 1 - (2*q.I*q.I + 2*q.K*q.K)
	m[5] = 2*q.J*q.K + 2*q.I*q.R
	m[6] = 2*q.I*q.K + 2*q.J*q.R
	m[7] = 2*q.J*q.K - 2*q.I*q.R
	m[8] = 1 - (2*q.I*q.I + 2*q.J*q.J)
}

func (m *Matrix3) LinearInterpolate(a, b *Matrix3, prop float64) *Matrix3 {
	result := &Matrix3{}
	for i := uint8(0); i < 9; i++ {
		result[i] = a[i]*(1-prop) + b[i]*prop
	}
	return result
}

type Matrix4 [12]float64

func (m *Matrix4) Clone() *Matrix4 {
	n := &Matrix4{}
	for i := range m {
		n[i] = m[i]
	}
	return n
}

func (m *Matrix4) Equals(other *Matrix4) bool {
	if other == nil {
		return false
	}
	for i := range m {
		if math.Abs(m[i]-other[i]) > RealEpsilon {
			return false
		}
	}
	return true
}

func (m *Matrix4) TransformVector3(b *Vector3) *Vector3 {
	return &Vector3{
		b[0]*m[0] + b[1]*m[1] + b[2]*m[2] + m[3],
		b[0]*m[4] + b[1]*m[5] + b[2]*m[6] + m[7],
		b[0]*m[8] + b[1]*m[9] + b[2]*m[10] + m[11],
	}
}

func (m *Matrix4) TransformMatrix4(b *Matrix4) *Matrix4 {
	newMatrix := &Matrix4{}

	newMatrix[0] = b[0]*m[0] + b[4]*m[1] + b[8]*m[2]
	newMatrix[4] = b[0]*m[4] + b[4]*m[5] + b[8]*m[6]
	newMatrix[8] = b[0]*m[8] + b[4]*m[9] + b[8]*m[10]

	newMatrix[1] = b[1]*m[0] + b[5]*m[1] + b[9]*m[2]
	newMatrix[5] = b[1]*m[4] + b[5]*m[5] + b[9]*m[6]
	newMatrix[9] = b[1]*m[8] + b[5]*m[9] + b[9]*m[10]

	newMatrix[2] = b[2]*m[0] + b[6]*m[1] + b[10]*m[2]
	newMatrix[6] = b[2]*m[4] + b[6]*m[5] + b[10]*m[6]
	newMatrix[10] = b[2]*m[8] + b[6]*m[9] + b[10]*m[10]

	newMatrix[3] = b[3]*m[0] + b[7]*m[1] + b[11]*m[2] + m[3]
	newMatrix[7] = b[3]*m[4] + b[7]*m[5] + b[11]*m[6] + m[7]
	newMatrix[11] = b[3]*m[8] + b[7]*m[9] + b[11]*m[10] + m[11]

	return newMatrix
}

func (m *Matrix4) getDeterminant() float64 {
	return m[8]*m[5]*m[2] + m[4]*m[9]*m[2] + m[8]*m[1]*m[6] - m[0]*m[9]*m[6] - m[4]*m[1]*m[10] + m[0]*m[5]*m[10]
}

// https://github.com/stojg/cyclone-physics/blob/master/src/core.cpp#L55
func (m *Matrix4) SetInverse(b *Matrix4) {
	det := m.getDeterminant()
	if det == 0 {
		return
	}

	det = 1.0 / det

	m[0] = (-b[9]*b[6] + b[5]*b[10]) * det
	m[4] = (b[8]*b[6] - b[4]*b[10]) * det
	m[8] = (-b[8]*b[5] + b[4]*b[9]) * det

	m[1] = (b[9]*b[2] - b[1]*b[10]) * det
	m[5] = (-b[8]*b[2] + b[0]*b[10]) * det
	m[9] = (b[8]*b[1] - b[0]*b[9]) * det

	m[2] = (-b[5]*b[2] + b[1]*b[6]) * det
	m[6] = (+b[4]*b[2] - b[0]*b[6]) * det
	m[10] = (-b[4]*b[1] + b[0]*b[5]) * det

	m[3] = (+b[9]*b[6]*b[3] - b[5]*b[10]*b[3] - b[9]*b[2]*b[7] + b[1]*b[10]*b[7] + b[5]*b[2]*b[11] - b[1]*b[6]*b[11]) * det
	m[7] = (-b[8]*b[6]*b[3] + b[4]*b[10]*b[3] + b[8]*b[2]*b[7] - b[0]*b[10]*b[7] - b[4]*b[2]*b[11] + b[0]*b[6]*b[11]) * det
	m[11] = (+b[8]*b[6]*b[3] - b[4]*b[9]*b[3] - b[8]*b[1]*b[7] + b[0]*b[9]*b[7] + b[4]*b[1]*b[11] - b[0]*b[5]*b[11]) * det
}

func (m *Matrix4) Inverse() *Matrix4 {
	result := &Matrix4{}
	result.SetInverse(m)
	return result
}

func (m *Matrix4) SetOrientation(q *Quaternion, b *Vector3) {
	m[0] = 1 - (2*q.J*q.J + 2*q.K*q.K)
	m[1] = 2*q.I*q.J + 2*q.K*q.R
	m[2] = 2*q.I*q.K - 2*q.J*q.R
	m[3] = b[0]

	m[4] = 2*q.I*q.J - 2*q.K*q.R
	m[5] = 1 - (2*q.I*q.I + 2*q.K*q.K)
	m[6] = 2*q.J*q.K + 2*q.I*q.R
	m[7] = b[1]

	m[8] = 2*q.I*q.K + 2*q.J*q.R
	m[9] = 2*q.J*q.K - 2*q.I*q.R
	m[10] = 1 - (2*q.I*q.I + 2*q.J*q.J)
	m[11] = b[2]
}

/**
 * Transform the given vector by the transformational inverse
 * of this matrix.
 *
 * @note This function relies on the fact that the inverse of
 * a pure rotation matrix is its transpose. It separates the
 * translational and rotation components, transposes the
 * rotation, and multiplies out. If the matrix is not a
 * scale and shear free transform matrix, then this function
 * will not give correct results.
 *
 * @param vector The vector to transform.
 */
func (m *Matrix4) TransformInverse(b *Vector3) *Vector3 {
	tmp := &Vector3{}
	tmp[0] -= m[3]
	tmp[1] -= m[7]
	tmp[2] -= m[11]

	result := &Vector3{}
	result[0] = tmp[0]*m[0] + tmp[1]*m[4] + tmp[2]*m[8]
	result[1] = tmp[0]*m[1] + tmp[1]*m[5] + tmp[2]*m[9]
	result[2] = tmp[0]*m[2] + tmp[1]*m[6] + tmp[2]*m[10]
	return result
}

func (m *Matrix4) TransformDirection(b *Vector3) *Vector3 {
	result := &Vector3{}
	result[0] = b[0]*m[0] + b[1]*m[1] + b[2]*m[2]
	result[1] = b[0]*m[4] + b[1]*m[5] + b[2]*m[6]
	result[2] = b[0]*m[8] + b[1]*m[9] + b[2]*m[10]
	return result
}

func (m *Matrix4) TransformInverseDirection(b *Vector3) *Vector3 {
	result := &Vector3{}
	result[0] = b[0]*m[0] + b[1]*m[4] + b[2]*m[8]
	result[1] = b[0]*m[1] + b[1]*m[5] + b[2]*m[9]
	result[2] = b[0]*m[2] + b[1]*m[6] + b[2]*m[10]
	return result
}

type Quaternion struct {
	R float64
	I float64
	J float64
	K float64
}

// zero rotation
func NewQuaternion(r, i, j, k float64) *Quaternion {
	return &Quaternion{r, i, j, k}
}

func QuaternionToTarget(origin, target *Vector3) *Quaternion {
	dest := target.NewSub(origin).Normalize()

	source := Z()
	dot := source.Dot(dest)
	if math.Abs(dot-(-1.0)) < RealEpsilon {
		// vector a and b point exactly in the opposite direction,
		// so it is a 180 degrees turn around the up-axis
		//return new Quaternion(up, MathHelper.ToRadians(180.0f));
		return QuaternionFromAxisAngle(Y(), -math.Pi)
	} else if math.Abs(dot-(1.0)) < RealEpsilon {
		// vector a and b point exactly in the same direction
		// so we return the identity quaternion
		return &Quaternion{1, 0, 0, 0}
	}
	rotAngle := math.Acos(dot)
	rotAxis := source.NewCross(dest).Normalize()
	return QuaternionFromAxisAngle(rotAxis, rotAngle)
}

func QuaternionFromAxisAngle(axis *Vector3, angle float64) *Quaternion {
	halfSin := math.Sin(angle / 2)
	halfCos := math.Cos(angle / 2)
	q := &Quaternion{
		halfCos,
		axis[0] * halfSin,
		axis[1] * halfSin,
		axis[2] * halfSin,
	}
	return q
}

func QuaternionFromVectors(a, b *Vector3) *Quaternion {

	m := math.Sqrt(2.0 + 2.0*a.Dot(b))

	w := a.Clone().NewCross(b).Scale(1.0 / m)

	return &Quaternion{
		0.5 * m,
		w[0],
		w[1],
		w[2],
	}
}

func (q *Quaternion) Set(r, i, j, k float64) {
	q.R = r
	q.I = i
	q.J = j
	q.K = k
}

func (q *Quaternion) Clone() *Quaternion {
	return &Quaternion{
		R: q.R,
		I: q.I,
		J: q.J,
		K: q.K,
	}
}

func (q *Quaternion) Equals(z *Quaternion) bool {
	if math.Abs(q.R-z.R) > RealEpsilon {
		return false
	}
	if math.Abs(q.I-z.I) > RealEpsilon {
		return false
	}
	if math.Abs(q.J-z.J) > RealEpsilon {
		return false
	}
	if math.Abs(q.K-z.K) > RealEpsilon {
		return false
	}
	return true
}

// Normalises the quaternion to unit length, making it a valid orientation quaternion.
func (q *Quaternion) Normalize() {
	d := q.R*q.R + q.I*q.I + q.J*q.J + q.K*q.K
	// Check for zero length quaternion, and use the no-rotation
	// quaternion in that case.
	if d < RealEpsilon {
		q.R = 1
		return
	}
	d = 1.0 / math.Sqrt(d)
	q.R *= d
	q.I *= d
	q.J *= d
	q.K *= d
}

// Conjugate transforms the quaternion by negating it's vector components:
// http://www.3dgep.com/understanding-quaternions/#Quaternion_Conjugate
func (q *Quaternion) Conjugate() *Quaternion {
	q.I = -q.I
	q.J = -q.J
	q.K = -q.K
	return q
}

// Conjugate returns a new conjugate version of this quaternion
func (q *Quaternion) NewConjugate() *Quaternion {
	return q.Clone().Conjugate()
}

func (q *Quaternion) NewInverse() *Quaternion {
	t := q.SquareLength()
	return &Quaternion{
		q.R / t,
		-q.I / t,
		-q.J / t,
		-q.K / t,
	}
}

func (q *Quaternion) Dot(q2 *Quaternion) float64 {
	return q.R*q2.R + q.I*q2.I + q.J*q2.J + q.K*q2.K
}

func (q *Quaternion) Div(s float64) *Quaternion {
	return &Quaternion{q.R / s, q.I / s, q.J / s, q.K / s}
}

func (q *Quaternion) Length() float64 {
	d := q.R*q.R + q.I*q.I + q.J*q.J + q.K*q.K
	return math.Sqrt(d)
}

func (q *Quaternion) SquareLength() float64 {
	return q.R*q.R + q.I*q.I + q.J*q.J + q.K*q.K
}

func (q *Quaternion) Norm() float64 {
	return q.SquareLength()
}

// Multiply multiplies the receiver quaternion by the given quaternion.
func (q *Quaternion) Multiply(o *Quaternion) *Quaternion {
	*q = Quaternion{
		-q.I*o.I - q.J*o.J - q.K*o.K + q.R*o.R,
		q.I*o.R + q.J*o.K - q.K*o.J + q.R*o.I,
		-q.I*o.K + q.J*o.R + q.K*o.I + q.R*o.J,
		q.I*o.J - q.J*o.I + q.K*o.R + q.R*o.K,
	}
	return q
}

// NewMultiply returns a new quaternion from the result of multiplying the receiver by the given
// quaternion.
func (q *Quaternion) NewMultiply(o *Quaternion) *Quaternion {
	return q.Clone().Multiply(o)
}

// AddScaledVector adds the given vector to this, scaled by the given amount. This can be use to
// update the orientation quaternion by a rotation and time.
func (q *Quaternion) AddScaledVector(vector *Vector3, scale float64) {

	vectorQ := &Quaternion{0, vector[0] * scale, vector[1] * scale, vector[2] * scale}
	result := vectorQ.NewMultiply(q)
	result.Div(2)

	q.R += result.R
	q.I += result.I
	q.J += result.J
	q.K += result.K
}

func (q *Quaternion) RotateByVector(vector *Vector3) *Quaternion {
	return q.Multiply(&Quaternion{0, vector[0], vector[1], vector[2]})
}

func (q *Quaternion) NewRotateByVector(vector *Vector3) *Quaternion {
	return q.NewMultiply(&Quaternion{0, vector[0], vector[1], vector[2]})
}

func LocalToWorld(local *Vector3, transform *Matrix4) *Vector3 {
	return transform.TransformVector3(local)
}

func WorldToLocal(world *Vector3, transform *Matrix4) *Vector3 {
	return transform.TransformInverse(world)
}

func LocalToWorldDirn(local *Vector3, transform *Matrix4) *Vector3 {
	return transform.TransformDirection(local)
}

func WorldToLocalDirn(world *Vector3, transform *Matrix4) *Vector3 {
	return transform.TransformInverseDirection(world)
}
