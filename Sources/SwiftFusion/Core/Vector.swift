// WARNING: This is a generated file. Do not edit it. Instead, edit the corresponding ".gyb" file.
// See "generate.sh" in the root of this repository for instructions how to regenerate files.

// ###sourceLocation(file: "Sources/SwiftFusion/Core/Vector.swift.gyb", line: 1)
import TensorFlow

public protocol FixedDimensionVector {
  static var dimension: Int { get }
  static var standardBasis: [Self] { get }
  var scalars: [Double] { get }
}

// ###sourceLocation(file: "Sources/SwiftFusion/Core/Vector.swift.gyb", line: 12)

/// An element of R^1, with Euclidean norm.
public struct Vector1: Differentiable, KeyPathIterable, TangentStandardBasis
{
// ###sourceLocation(file: "Sources/SwiftFusion/Core/Vector.swift.gyb", line: 17)
  @differentiable public var x: Double
// ###sourceLocation(file: "Sources/SwiftFusion/Core/Vector.swift.gyb", line: 19)

  @differentiable
  public init(_ x: Double) {
// ###sourceLocation(file: "Sources/SwiftFusion/Core/Vector.swift.gyb", line: 23)
    self.x = x
// ###sourceLocation(file: "Sources/SwiftFusion/Core/Vector.swift.gyb", line: 25)
  }
}

/// Normed vector space methods.
extension Vector1: AdditiveArithmetic, VectorProtocol {
  public typealias VectorSpaceScalar = Double

  /// Euclidean norm of `self`.
  @differentiable
  public var norm: Double { squaredNorm.squareRoot() }

  /// Square of the Euclidean norm of `self`.
  @differentiable
  public var squaredNorm: Double { self.squared().sum() }

  @differentiable
  public static func + (_ lhs: Self, _ rhs: Self) -> Self {
    var result = Self.zero
// ###sourceLocation(file: "Sources/SwiftFusion/Core/Vector.swift.gyb", line: 44)
    result.x = lhs.x + rhs.x
// ###sourceLocation(file: "Sources/SwiftFusion/Core/Vector.swift.gyb", line: 46)
    return result
  }

  @differentiable
  public static func - (_ lhs: Self, _ rhs: Self) -> Self {
    var result = Self.zero
// ###sourceLocation(file: "Sources/SwiftFusion/Core/Vector.swift.gyb", line: 53)
    result.x = lhs.x - rhs.x
// ###sourceLocation(file: "Sources/SwiftFusion/Core/Vector.swift.gyb", line: 55)
    return result
  }

  @differentiable
  public static prefix func - (_ v: Self) -> Self {
    var result = Self.zero
// ###sourceLocation(file: "Sources/SwiftFusion/Core/Vector.swift.gyb", line: 62)
    result.x = -v.x
// ###sourceLocation(file: "Sources/SwiftFusion/Core/Vector.swift.gyb", line: 64)
    return result
  }
}

/// Other arithmetic on the vector elements.
extension Vector1: ElementaryFunctions {
  /// Sum of the elements of `self`.
  public func sum() -> Double {
    var result: Double = 0
// ###sourceLocation(file: "Sources/SwiftFusion/Core/Vector.swift.gyb", line: 74)
    result = result + x
// ###sourceLocation(file: "Sources/SwiftFusion/Core/Vector.swift.gyb", line: 76)
    return result
  }

  /// Vector whose elements are squares of the elements of `self`.
  public func squared() -> Self {
    var result = Self.zero
// ###sourceLocation(file: "Sources/SwiftFusion/Core/Vector.swift.gyb", line: 83)
    result.x = x * x
// ###sourceLocation(file: "Sources/SwiftFusion/Core/Vector.swift.gyb", line: 85)
    return result
  }
}

/// Conformance to `FixedDimensionVector`.
extension Vector1: FixedDimensionVector {
  public static var dimension: Int { return 1 }

  public static var standardBasis: [Self] {
// ###sourceLocation(file: "Sources/SwiftFusion/Core/Vector.swift.gyb", line: 95)
    var basisx: Self = .zero
    basisx.x = 1
// ###sourceLocation(file: "Sources/SwiftFusion/Core/Vector.swift.gyb", line: 98)
    return [basisx]
  }

  public var scalars: [Double] {
    return [x]
  }
}

/// Conversion to/from tensor.
extension Vector1 {
  /// A `Tensor` with shape `[1]` whose elements are the elements of `self`.
  @differentiable
  public var tensor: Tensor<Double> {
    Tensor([x])
  }

  /// Creates a `Vector1` with the same elements as `tensor`.
  ///
  /// Precondition: `tensor` must have shape `[1]`.
  @differentiable
  public init(_ tensor: Tensor<Double>) {
    precondition(tensor.shape == [1])
// ###sourceLocation(file: "Sources/SwiftFusion/Core/Vector.swift.gyb", line: 121)
    self.x = tensor[0].scalarized()
// ###sourceLocation(file: "Sources/SwiftFusion/Core/Vector.swift.gyb", line: 123)
  }
}

// ###sourceLocation(file: "Sources/SwiftFusion/Core/Vector.swift.gyb", line: 12)

/// An element of R^2, with Euclidean norm.
public struct Vector2: Differentiable, KeyPathIterable, TangentStandardBasis
{
// ###sourceLocation(file: "Sources/SwiftFusion/Core/Vector.swift.gyb", line: 17)
  @differentiable public var x: Double
// ###sourceLocation(file: "Sources/SwiftFusion/Core/Vector.swift.gyb", line: 17)
  @differentiable public var y: Double
// ###sourceLocation(file: "Sources/SwiftFusion/Core/Vector.swift.gyb", line: 19)

  @differentiable
  public init(_ x: Double, _ y: Double) {
// ###sourceLocation(file: "Sources/SwiftFusion/Core/Vector.swift.gyb", line: 23)
    self.x = x
// ###sourceLocation(file: "Sources/SwiftFusion/Core/Vector.swift.gyb", line: 23)
    self.y = y
// ###sourceLocation(file: "Sources/SwiftFusion/Core/Vector.swift.gyb", line: 25)
  }
}

/// Normed vector space methods.
extension Vector2: AdditiveArithmetic, VectorProtocol {
  public typealias VectorSpaceScalar = Double

  /// Euclidean norm of `self`.
  @differentiable
  public var norm: Double { squaredNorm.squareRoot() }

  /// Square of the Euclidean norm of `self`.
  @differentiable
  public var squaredNorm: Double { self.squared().sum() }

  @differentiable
  public static func + (_ lhs: Self, _ rhs: Self) -> Self {
    var result = Self.zero
// ###sourceLocation(file: "Sources/SwiftFusion/Core/Vector.swift.gyb", line: 44)
    result.x = lhs.x + rhs.x
// ###sourceLocation(file: "Sources/SwiftFusion/Core/Vector.swift.gyb", line: 44)
    result.y = lhs.y + rhs.y
// ###sourceLocation(file: "Sources/SwiftFusion/Core/Vector.swift.gyb", line: 46)
    return result
  }

  @differentiable
  public static func - (_ lhs: Self, _ rhs: Self) -> Self {
    var result = Self.zero
// ###sourceLocation(file: "Sources/SwiftFusion/Core/Vector.swift.gyb", line: 53)
    result.x = lhs.x - rhs.x
// ###sourceLocation(file: "Sources/SwiftFusion/Core/Vector.swift.gyb", line: 53)
    result.y = lhs.y - rhs.y
// ###sourceLocation(file: "Sources/SwiftFusion/Core/Vector.swift.gyb", line: 55)
    return result
  }

  @differentiable
  public static prefix func - (_ v: Self) -> Self {
    var result = Self.zero
// ###sourceLocation(file: "Sources/SwiftFusion/Core/Vector.swift.gyb", line: 62)
    result.x = -v.x
// ###sourceLocation(file: "Sources/SwiftFusion/Core/Vector.swift.gyb", line: 62)
    result.y = -v.y
// ###sourceLocation(file: "Sources/SwiftFusion/Core/Vector.swift.gyb", line: 64)
    return result
  }
}

/// Other arithmetic on the vector elements.
extension Vector2: ElementaryFunctions {
  /// Sum of the elements of `self`.
  public func sum() -> Double {
    var result: Double = 0
// ###sourceLocation(file: "Sources/SwiftFusion/Core/Vector.swift.gyb", line: 74)
    result = result + x
// ###sourceLocation(file: "Sources/SwiftFusion/Core/Vector.swift.gyb", line: 74)
    result = result + y
// ###sourceLocation(file: "Sources/SwiftFusion/Core/Vector.swift.gyb", line: 76)
    return result
  }

  /// Vector whose elements are squares of the elements of `self`.
  public func squared() -> Self {
    var result = Self.zero
// ###sourceLocation(file: "Sources/SwiftFusion/Core/Vector.swift.gyb", line: 83)
    result.x = x * x
// ###sourceLocation(file: "Sources/SwiftFusion/Core/Vector.swift.gyb", line: 83)
    result.y = y * y
// ###sourceLocation(file: "Sources/SwiftFusion/Core/Vector.swift.gyb", line: 85)
    return result
  }
}

/// Conformance to `FixedDimensionVector`.
extension Vector2: FixedDimensionVector {
  public static var dimension: Int { return 2 }

  public static var standardBasis: [Self] {
// ###sourceLocation(file: "Sources/SwiftFusion/Core/Vector.swift.gyb", line: 95)
    var basisx: Self = .zero
    basisx.x = 1
// ###sourceLocation(file: "Sources/SwiftFusion/Core/Vector.swift.gyb", line: 95)
    var basisy: Self = .zero
    basisy.y = 1
// ###sourceLocation(file: "Sources/SwiftFusion/Core/Vector.swift.gyb", line: 98)
    return [basisx, basisy]
  }

  public var scalars: [Double] {
    return [x, y]
  }
}

/// Conversion to/from tensor.
extension Vector2 {
  /// A `Tensor` with shape `[2]` whose elements are the elements of `self`.
  @differentiable
  public var tensor: Tensor<Double> {
    Tensor([x, y])
  }

  /// Creates a `Vector2` with the same elements as `tensor`.
  ///
  /// Precondition: `tensor` must have shape `[2]`.
  @differentiable
  public init(_ tensor: Tensor<Double>) {
    precondition(tensor.shape == [2])
// ###sourceLocation(file: "Sources/SwiftFusion/Core/Vector.swift.gyb", line: 121)
    self.x = tensor[0].scalarized()
// ###sourceLocation(file: "Sources/SwiftFusion/Core/Vector.swift.gyb", line: 121)
    self.y = tensor[1].scalarized()
// ###sourceLocation(file: "Sources/SwiftFusion/Core/Vector.swift.gyb", line: 123)
  }
}

// ###sourceLocation(file: "Sources/SwiftFusion/Core/Vector.swift.gyb", line: 12)

/// An element of R^3, with Euclidean norm.
public struct Vector3: Differentiable, KeyPathIterable, TangentStandardBasis
{
// ###sourceLocation(file: "Sources/SwiftFusion/Core/Vector.swift.gyb", line: 17)
  @differentiable public var x: Double
// ###sourceLocation(file: "Sources/SwiftFusion/Core/Vector.swift.gyb", line: 17)
  @differentiable public var y: Double
// ###sourceLocation(file: "Sources/SwiftFusion/Core/Vector.swift.gyb", line: 17)
  @differentiable public var z: Double
// ###sourceLocation(file: "Sources/SwiftFusion/Core/Vector.swift.gyb", line: 19)

  @differentiable
  public init(_ x: Double, _ y: Double, _ z: Double) {
// ###sourceLocation(file: "Sources/SwiftFusion/Core/Vector.swift.gyb", line: 23)
    self.x = x
// ###sourceLocation(file: "Sources/SwiftFusion/Core/Vector.swift.gyb", line: 23)
    self.y = y
// ###sourceLocation(file: "Sources/SwiftFusion/Core/Vector.swift.gyb", line: 23)
    self.z = z
// ###sourceLocation(file: "Sources/SwiftFusion/Core/Vector.swift.gyb", line: 25)
  }
}

/// Normed vector space methods.
extension Vector3: AdditiveArithmetic, VectorProtocol {
  public typealias VectorSpaceScalar = Double

  /// Euclidean norm of `self`.
  @differentiable
  public var norm: Double { squaredNorm.squareRoot() }

  /// Square of the Euclidean norm of `self`.
  @differentiable
  public var squaredNorm: Double { self.squared().sum() }

  @differentiable
  public static func + (_ lhs: Self, _ rhs: Self) -> Self {
    var result = Self.zero
// ###sourceLocation(file: "Sources/SwiftFusion/Core/Vector.swift.gyb", line: 44)
    result.x = lhs.x + rhs.x
// ###sourceLocation(file: "Sources/SwiftFusion/Core/Vector.swift.gyb", line: 44)
    result.y = lhs.y + rhs.y
// ###sourceLocation(file: "Sources/SwiftFusion/Core/Vector.swift.gyb", line: 44)
    result.z = lhs.z + rhs.z
// ###sourceLocation(file: "Sources/SwiftFusion/Core/Vector.swift.gyb", line: 46)
    return result
  }

  @differentiable
  public static func - (_ lhs: Self, _ rhs: Self) -> Self {
    var result = Self.zero
// ###sourceLocation(file: "Sources/SwiftFusion/Core/Vector.swift.gyb", line: 53)
    result.x = lhs.x - rhs.x
// ###sourceLocation(file: "Sources/SwiftFusion/Core/Vector.swift.gyb", line: 53)
    result.y = lhs.y - rhs.y
// ###sourceLocation(file: "Sources/SwiftFusion/Core/Vector.swift.gyb", line: 53)
    result.z = lhs.z - rhs.z
// ###sourceLocation(file: "Sources/SwiftFusion/Core/Vector.swift.gyb", line: 55)
    return result
  }

  @differentiable
  public static prefix func - (_ v: Self) -> Self {
    var result = Self.zero
// ###sourceLocation(file: "Sources/SwiftFusion/Core/Vector.swift.gyb", line: 62)
    result.x = -v.x
// ###sourceLocation(file: "Sources/SwiftFusion/Core/Vector.swift.gyb", line: 62)
    result.y = -v.y
// ###sourceLocation(file: "Sources/SwiftFusion/Core/Vector.swift.gyb", line: 62)
    result.z = -v.z
// ###sourceLocation(file: "Sources/SwiftFusion/Core/Vector.swift.gyb", line: 64)
    return result
  }
}

/// Other arithmetic on the vector elements.
extension Vector3: ElementaryFunctions {
  /// Sum of the elements of `self`.
  public func sum() -> Double {
    var result: Double = 0
// ###sourceLocation(file: "Sources/SwiftFusion/Core/Vector.swift.gyb", line: 74)
    result = result + x
// ###sourceLocation(file: "Sources/SwiftFusion/Core/Vector.swift.gyb", line: 74)
    result = result + y
// ###sourceLocation(file: "Sources/SwiftFusion/Core/Vector.swift.gyb", line: 74)
    result = result + z
// ###sourceLocation(file: "Sources/SwiftFusion/Core/Vector.swift.gyb", line: 76)
    return result
  }

  /// Vector whose elements are squares of the elements of `self`.
  public func squared() -> Self {
    var result = Self.zero
// ###sourceLocation(file: "Sources/SwiftFusion/Core/Vector.swift.gyb", line: 83)
    result.x = x * x
// ###sourceLocation(file: "Sources/SwiftFusion/Core/Vector.swift.gyb", line: 83)
    result.y = y * y
// ###sourceLocation(file: "Sources/SwiftFusion/Core/Vector.swift.gyb", line: 83)
    result.z = z * z
// ###sourceLocation(file: "Sources/SwiftFusion/Core/Vector.swift.gyb", line: 85)
    return result
  }
}

/// Conformance to `FixedDimensionVector`.
extension Vector3: FixedDimensionVector {
  public static var dimension: Int { return 3 }

  public static var standardBasis: [Self] {
// ###sourceLocation(file: "Sources/SwiftFusion/Core/Vector.swift.gyb", line: 95)
    var basisx: Self = .zero
    basisx.x = 1
// ###sourceLocation(file: "Sources/SwiftFusion/Core/Vector.swift.gyb", line: 95)
    var basisy: Self = .zero
    basisy.y = 1
// ###sourceLocation(file: "Sources/SwiftFusion/Core/Vector.swift.gyb", line: 95)
    var basisz: Self = .zero
    basisz.z = 1
// ###sourceLocation(file: "Sources/SwiftFusion/Core/Vector.swift.gyb", line: 98)
    return [basisx, basisy, basisz]
  }

  public var scalars: [Double] {
    return [x, y, z]
  }
}

/// Conversion to/from tensor.
extension Vector3 {
  /// A `Tensor` with shape `[3]` whose elements are the elements of `self`.
  @differentiable
  public var tensor: Tensor<Double> {
    Tensor([x, y, z])
  }

  /// Creates a `Vector3` with the same elements as `tensor`.
  ///
  /// Precondition: `tensor` must have shape `[3]`.
  @differentiable
  public init(_ tensor: Tensor<Double>) {
    precondition(tensor.shape == [3])
// ###sourceLocation(file: "Sources/SwiftFusion/Core/Vector.swift.gyb", line: 121)
    self.x = tensor[0].scalarized()
// ###sourceLocation(file: "Sources/SwiftFusion/Core/Vector.swift.gyb", line: 121)
    self.y = tensor[1].scalarized()
// ###sourceLocation(file: "Sources/SwiftFusion/Core/Vector.swift.gyb", line: 121)
    self.z = tensor[2].scalarized()
// ###sourceLocation(file: "Sources/SwiftFusion/Core/Vector.swift.gyb", line: 123)
  }
}

