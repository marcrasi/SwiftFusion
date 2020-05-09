// Copyright 2019 The SwiftFusion Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
import TensorFlow

/// The most general factor protocol.
public protocol Factor {
  var keys: Array<Int> { get }
}

/// A `NonlinearFactor` corresponds to the `NonlinearFactor` in GTSAM.
///
/// Input is a dictionary of `Key` to `Value` pairs, and the output is the scalar
/// error value
///
/// Interpretation
/// ================
/// `Input`: the input values as key-value pairs
///
public protocol NonlinearFactor: Factor {
  typealias ScalarType = Double
  
  /// TODO: `Dictionary` still does not conform to `Differentiable`
  /// Tracking issue: https://bugs.swift.org/browse/TF-899
//  typealias Input = Dictionary<UInt, Tensor<ScalarType>>
  
  /// Returns the `error` of the factor.
  @differentiable(wrt: values)
  func error(_ values: Values) -> ScalarType
  
  func linearization(_ values: Values) -> (linearMap: SparseMatrix, bias: Vector)
}
