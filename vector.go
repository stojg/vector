package vector

import (
	"fmt"
	"math"
)

type Matrix3 [9]float64

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

// http://pastebin.com/fAFp6NnN
func (value *Vector3) Rotate(rotation *Quaternion) *Vector3 {
	num12 := rotation.I + rotation.I
	num2 := rotation.J + rotation.J
	num := rotation.K + rotation.K
	num11 := rotation.R * num12
	num10 := rotation.R * num2
	num9 := rotation.R * num
	num8 := rotation.I * num12
	num7 := rotation.I * num2
	num6 := rotation.I * num
	num5 := rotation.J * num2
	num4 := rotation.J * num
	num3 := rotation.K * num
	num15 := ((value[0] * ((1.0 - num5) - num3)) + (value[1] * (num7 - num9))) + (value[2] * (num6 + num10))
	num14 := ((value[0] * (num7 + num9)) + (value[1] * ((1.0 - num8) - num3))) + (value[2] * (num4 - num11))
	num13 := ((value[0] * (num6 - num10)) + (value[1] * (num4 + num11))) + (value[2] * ((1.0 - num8) - num5))

	value[0] = num15
	value[1] = num14
	value[2] = num13
	return value
}

func (a *Vector3) NewRotate(q *Quaternion) *Vector3 {
	return a.Clone().Rotate(q)
}

func (data *Matrix3) TransformVector3(vector *Vector3) *Vector3 {
	return &Vector3{
		vector[0]*data[0] + vector[1]*data[1] + vector[2]*data[2],
		vector[0]*data[3] + vector[1]*data[4] + vector[2]*data[5],
		vector[0]*data[6] + vector[1]*data[7] + vector[2]*data[8],
	}
}

func (m *Matrix3) TransformMatrix3(o *Matrix3) *Matrix3 {
	newMatrix := &Matrix3{}
	newMatrix[0] = m[0]*o[0] + m[1]*o[3] + m[2] + o[6]
	newMatrix[1] = m[0]*o[1] + m[1]*o[4] + m[2] + o[7]
	newMatrix[2] = m[0]*o[2] + m[1]*o[5] + m[2] + o[8]
	newMatrix[3] = m[3]*o[0] + m[4]*o[3] + m[5] + o[6]
	newMatrix[4] = m[3]*o[1] + m[4]*o[5] + m[5] + o[7]
	newMatrix[5] = m[3]*o[2] + m[4]*o[6] + m[5] + o[8]
	newMatrix[6] = m[6]*o[0] + m[7]*o[3] + m[8] + o[6]
	newMatrix[7] = m[6]*o[1] + m[7]*o[4] + m[8] + o[7]
	newMatrix[8] = m[6]*o[2] + m[7]*o[5] + m[8] + o[8]
	return newMatrix
}

func (a *Matrix3) SetInverse(b *Matrix3) {

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

	a[0] = (b[4]*b[8] - b[5]*b[7]) * invd
	a[1] = -(b[1]*b[8] - b[2]*b[7]) * invd
	a[2] = (b[1]*b[5] - b[2]*b[4]) * invd

	a[3] = -(b[3]*b[8] - b[5]*b[6]) * invd
	a[4] = (b[0]*b[8] - t6) * invd
	a[5] = -(t2 - t4) * invd

	a[6] = (b[3]*b[7] - b[4]*b[6]) * invd
	a[7] = -(b[0]*b[7] - t5) * invd
	a[8] = (t1 - t3) * invd

}

// Returns a new matrix containing the inverse of this matrix
func (m *Matrix3) Inverse() *Matrix3 {
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

func (orig *Matrix3) SetTranspose(m *Matrix3) {
	orig[0] = m[0]
	orig[1] = m[3]
	orig[2] = m[6]
	orig[3] = m[1]
	orig[4] = m[4]
	orig[5] = m[7]
	orig[6] = m[2]
	orig[7] = m[5]
	orig[8] = m[8]
}

func (orig *Matrix3) Transpose(m *Matrix3) *Matrix3 {
	result := &Matrix3{}
	result.SetTranspose(orig)
	return result
}

func (data *Matrix3) SetOrientation(q *Quaternion) {
	data[0] = 1 - (2*q.J*q.J + 2*q.K*q.K)
	data[1] = 2*q.I*q.J + 2*q.K*q.R
	data[2] = 2*q.I*q.K - 2*q.J*q.R
	data[3] = 2*q.I*q.J - 2*q.K*q.R
	data[4] = 1 - (2*q.I*q.I + 2*q.K*q.K)
	data[5] = 2*q.J*q.K + 2*q.I*q.R
	data[6] = 2*q.I*q.K + 2*q.J*q.R
	data[7] = 2*q.J*q.K - 2*q.I*q.R
	data[8] = 1 - (2*q.I*q.I + 2*q.J*q.J)
}

func (data *Matrix3) SetOrientationAndPos(q *Quaternion, pos *Vector3) {

}

func (m *Matrix3) LinearInterpolate(a, b *Matrix3, prop float64) *Matrix3 {
	result := &Matrix3{}
	for i := uint8(0); i < 9; i++ {
		result[i] = a[i]*(1-prop) + b[i]*prop
	}
	return result
}

type Matrix4 [12]float64

func (m *Matrix4) TransformVector3(v *Vector3) *Vector3 {
	return &Vector3{
		v[0]*m[0] + v[1]*m[1] + v[2]*m[2] + m[3],
		v[0]*m[4] + v[1]*m[5] + v[2]*m[6] + m[7],
		v[0]*m[8] + v[1]*m[9] + v[2]*m[10] + m[11],
	}
}

func (m *Matrix4) TransformMatrix4(o *Matrix4) *Matrix4 {
	newMatrix := &Matrix4{}

	newMatrix[0] = o[0]*m[0] + o[4]*m[1] + o[8]*m[2]
	newMatrix[4] = o[0]*m[4] + o[4]*m[5] + o[8]*m[6]
	newMatrix[8] = o[0]*m[8] + o[4]*m[9] + o[8]*m[10]

	newMatrix[1] = o[1]*m[0] + o[5]*m[1] + o[9]*m[2]
	newMatrix[5] = o[1]*m[4] + o[5]*m[5] + o[9]*m[6]
	newMatrix[9] = o[1]*m[8] + o[5]*m[9] + o[9]*m[10]

	newMatrix[2] = o[2]*m[0] + o[6]*m[1] + o[10]*m[2]
	newMatrix[6] = o[2]*m[4] + o[6]*m[5] + o[10]*m[6]
	newMatrix[10] = o[2]*m[8] + o[6]*m[9] + o[10]*m[10]

	newMatrix[3] = o[3]*m[0] + o[7]*m[1] + o[11]*m[2] + m[3]
	newMatrix[7] = o[3]*m[4] + o[7]*m[5] + o[11]*m[6] + m[7]
	newMatrix[11] = o[3]*m[8] + o[7]*m[9] + o[11]*m[10] + m[11]

	return newMatrix
}

func (m *Matrix4) getDeterminant() float64 {
	return m[8]*m[5]*m[2] + m[4]*m[9]*m[2] + m[8]*m[1]*m[6] - m[0]*m[9]*m[6] - m[4]*m[1]*m[10] + m[0]*m[5]*m[10]
}

// https://github.com/stojg/cyclone-physics/blob/master/src/core.cpp#L55
func (data *Matrix4) SetInverse(m *Matrix4) {
	det := data.getDeterminant()
	if det == 0 {
		return
	}

	det = 1.0 / det

	data[0] = (-m[9]*m[6] + m[5]*m[10]) * det
	data[4] = (m[8]*m[6] - m[4]*m[10]) * det
	data[8] = (-m[8]*m[5] + m[4]*m[9]) * det

	data[1] = (m[9]*m[2] - m[1]*m[10]) * det
	data[5] = (-m[8]*m[2] + m[0]*m[10]) * det
	data[9] = (m[8]*m[1] - m[0]*m[9]) * det

	data[2] = (-m[5]*m[2] + m[1]*m[6]) * det
	data[6] = (+m[4]*m[2] - m[0]*m[6]) * det
	data[10] = (-m[4]*m[1] + m[0]*m[5]) * det

	data[3] = (+m[9]*m[6]*m[3] - m[5]*m[10]*m[3] - m[9]*m[2]*m[7] + m[1]*m[10]*m[7] + m[5]*m[2]*m[11] - m[1]*m[6]*m[11]) * det
	data[7] = (-m[8]*m[6]*m[3] + m[4]*m[10]*m[3] + m[8]*m[2]*m[7] - m[0]*m[10]*m[7] - m[4]*m[2]*m[11] + m[0]*m[6]*m[11]) * det
	data[11] = (+m[8]*m[6]*m[3] - m[4]*m[9]*m[3] - m[8]*m[1]*m[7] + m[0]*m[9]*m[7] + m[4]*m[1]*m[11] - m[0]*m[5]*m[11]) * det
}

func (m *Matrix4) Inverse() *Matrix4 {
	result := &Matrix4{}
	result.SetInverse(m)
	return result
}

func (data *Matrix4) SetOrientation(q *Quaternion, pos *Vector3) {
	data[0] = 1 - (2*q.J*q.J + 2*q.K*q.K)
	data[1] = 2*q.I*q.J + 2*q.K*q.R
	data[2] = 2*q.I*q.K - 2*q.J*q.R
	data[3] = pos[0]

	data[4] = 2*q.I*q.J - 2*q.K*q.R
	data[5] = 1 - (2*q.I*q.I + 2*q.K*q.K)
	data[6] = 2*q.J*q.K + 2*q.I*q.R
	data[7] = pos[1]

	data[8] = 2*q.I*q.K + 2*q.J*q.R
	data[9] = 2*q.J*q.K - 2*q.I*q.R
	data[10] = 1 - (2*q.I*q.I + 2*q.J*q.J)
	data[11] = pos[2]
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
func (data *Matrix4) TransformInverse(vector *Vector3) *Vector3 {
	tmp := &Vector3{}
	tmp[0] -= data[3]
	tmp[1] -= data[7]
	tmp[2] -= data[11]

	result := &Vector3{}
	result[0] = tmp[0]*data[0] + tmp[1]*data[4] + tmp[2]*data[8]
	result[1] = tmp[0]*data[1] + tmp[1]*data[5] + tmp[2]*data[9]
	result[2] = tmp[0]*data[2] + tmp[1]*data[6] + tmp[2]*data[10]
	return result
}

func (data *Matrix4) TransformDirection(vector *Vector3) *Vector3 {
	result := &Vector3{}
	result[0] = vector[0]*data[0] + vector[1]*data[1] + vector[2]*data[2]
	result[1] = vector[0]*data[4] + vector[1]*data[5] + vector[2]*data[6]
	result[2] = vector[0]*data[8] + vector[1]*data[9] + vector[2]*data[10]
	return result
}

func (data *Matrix4) TransformInverseDirection(vector *Vector3) *Vector3 {
	result := &Vector3{}
	result[0] = vector[0]*data[0] + vector[1]*data[4] + vector[2]*data[8]
	result[1] = vector[0]*data[1] + vector[1]*data[5] + vector[2]*data[9]
	result[2] = vector[0]*data[2] + vector[1]*data[6] + vector[2]*data[10]
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

	source := VectorZ()
	dot := source.Dot(dest)
	if math.Abs(dot-(-1.0)) < RealEpsilon {
		// vector a and b point exactly in the opposite direction,
		// so it is a 180 degrees turn around the up-axis
		//return new Quaternion(up, MathHelper.ToRadians(180.0f));
		return QuaternionFromAxisAngle(VectorY(), -math.Pi)
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

// http://www.ncsa.illinois.edu/People/kindr/emtc/quaternions/quaternion.c++
func (q *Quaternion) Conjugate() *Quaternion {
	q.I = -q.I
	q.J = -q.J
	q.K = -q.K
	return q
}

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

// Multiplies the quaternion by the given quaternion.
func (q *Quaternion) Multiply(o *Quaternion) *Quaternion {
	*q = Quaternion{
		-q.I*o.I - q.J*o.J - q.K*o.K + q.R*o.R,
		q.I*o.R + q.J*o.K - q.K*o.J + q.R*o.I,
		-q.I*o.K + q.J*o.R + q.K*o.I + q.R*o.J,
		q.I*o.J - q.J*o.I + q.K*o.R + q.R*o.K,
	}
	return q
}

// Multiplies the quaternion by the given quaternion.
func (q *Quaternion) NewMultiply(o *Quaternion) *Quaternion {
	return q.Clone().Multiply(o)
}

// Adds the given vector to this, scaled by the given amount. This is
// used to update the orientation quaternion by a rotation and time.
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

func (q *Quaternion) AsMatrix() *Matrix4 {
	return &Matrix4{}
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
