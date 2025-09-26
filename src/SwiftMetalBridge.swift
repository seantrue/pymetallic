// SwiftMetalBridge.swift
// Swift bridge library for Python Metal bindings
// Compile with: swiftc -emit-library -o libpymetal.dylib SwiftMetalBridge.swift

import Metal
import Foundation

// MARK: - Device Functions

@_cdecl("metal_get_default_device")
public func metal_get_default_device() -> UnsafeMutableRawPointer? {
    guard let device = MTLCreateSystemDefaultDevice() else {
        return nil
    }
    return Unmanaged.passRetained(device).toOpaque()
}

@_cdecl("metal_get_all_devices")
public func metal_get_all_devices() -> UnsafeMutablePointer<UnsafeMutableRawPointer?>? {
    let devices = MTLCopyAllDevices()
    guard !devices.isEmpty else {
        return nil
    }
    
    let devicePointers = UnsafeMutablePointer<UnsafeMutableRawPointer?>.allocate(capacity: devices.count)
    for (index, device) in devices.enumerated() {
        devicePointers[index] = Unmanaged.passRetained(device).toOpaque()
    }
    return devicePointers
}

@_cdecl("metal_get_device_count")
public func metal_get_device_count() -> Int32 {
    return Int32(MTLCopyAllDevices().count)
}

@_cdecl("metal_device_get_name")
public func metal_device_get_name(_ devicePtr: UnsafeMutableRawPointer) -> UnsafePointer<CChar>? {
    let device = Unmanaged<MTLDevice>.fromOpaque(devicePtr).takeUnretainedValue()
    let name = device.name
    return UnsafePointer(strdup(name))
}

@_cdecl("metal_device_supports_shader_barycentric_coordinates")
public func metal_device_supports_shader_barycentric_coordinates(_ devicePtr: UnsafeMutableRawPointer) -> Bool {
    let device = Unmanaged<MTLDevice>.fromOpaque(devicePtr).takeUnretainedValue()
    return device.supportsShaderBarycentricCoordinates
}

// MARK: - Command Queue Functions

@_cdecl("metal_device_make_command_queue")
public func metal_device_make_command_queue(_ devicePtr: UnsafeMutableRawPointer) -> UnsafeMutableRawPointer? {
    let device = Unmanaged<MTLDevice>.fromOpaque(devicePtr).takeUnretainedValue()
    guard let commandQueue = device.makeCommandQueue() else {
        return nil
    }
    return Unmanaged.passRetained(commandQueue).toOpaque()
}

@_cdecl("metal_command_queue_make_command_buffer")
public func metal_command_queue_make_command_buffer(_ queuePtr: UnsafeMutableRawPointer) -> UnsafeMutableRawPointer? {
    let commandQueue = Unmanaged<MTLCommandQueue>.fromOpaque(queuePtr).takeUnretainedValue()
    guard let commandBuffer = commandQueue.makeCommandBuffer() else {
        return nil
    }
    return Unmanaged.passRetained(commandBuffer).toOpaque()
}

// MARK: - Buffer Functions

@_cdecl("metal_device_make_buffer")
public func metal_device_make_buffer(_ devicePtr: UnsafeMutableRawPointer, 
                                   _ length: UInt64, 
                                   _ options: Int32) -> UnsafeMutableRawPointer? {
    let device = Unmanaged<MTLDevice>.fromOpaque(devicePtr).takeUnretainedValue()
    let resourceOptions = MTLResourceOptions(rawValue: UInt(options))
    
    guard let buffer = device.makeBuffer(length: Int(length), options: resourceOptions) else {
        return nil
    }
    return Unmanaged.passRetained(buffer).toOpaque()
}

@_cdecl("metal_device_make_buffer_with_bytes")
public func metal_device_make_buffer_with_bytes(_ devicePtr: UnsafeMutableRawPointer,
                                              _ bytes: UnsafeMutableRawPointer,
                                              _ length: UInt64,
                                              _ options: Int32) -> UnsafeMutableRawPointer? {
    let device = Unmanaged<MTLDevice>.fromOpaque(devicePtr).takeUnretainedValue()
    let resourceOptions = MTLResourceOptions(rawValue: UInt(options))
    
    guard let buffer = device.makeBuffer(bytes: bytes, length: Int(length), options: resourceOptions) else {
        return nil
    }
    return Unmanaged.passRetained(buffer).toOpaque()
}

@_cdecl("metal_buffer_get_contents")
public func metal_buffer_get_contents(_ bufferPtr: UnsafeMutableRawPointer) -> UnsafeMutableRawPointer? {
    let buffer = Unmanaged<MTLBuffer>.fromOpaque(bufferPtr).takeUnretainedValue()
    return buffer.contents()
}

@_cdecl("metal_buffer_get_length")
public func metal_buffer_get_length(_ bufferPtr: UnsafeMutableRawPointer) -> UInt64 {
    let buffer = Unmanaged<MTLBuffer>.fromOpaque(bufferPtr).takeUnretainedValue()
    return UInt64(buffer.length)
}

// MARK: - Library and Function Functions

@_cdecl("metal_device_make_library_with_source")
public func metal_device_make_library_with_source(_ devicePtr: UnsafeMutableRawPointer,
                                                _ source: UnsafePointer<CChar>) -> UnsafeMutableRawPointer? {
    let device = Unmanaged<MTLDevice>.fromOpaque(devicePtr).takeUnretainedValue()
    let sourceString = String(cString: source)
    
    do {
        let library = try device.makeLibrary(source: sourceString, options: nil)
        return Unmanaged.passRetained(library).toOpaque()
    } catch {
        print("Error compiling Metal library: \(error)")
        return nil
    }
}

@_cdecl("metal_library_make_function")
public func metal_library_make_function(_ libraryPtr: UnsafeMutableRawPointer,
                                      _ name: UnsafePointer<CChar>) -> UnsafeMutableRawPointer? {
    let library = Unmanaged<MTLLibrary>.fromOpaque(libraryPtr).takeUnretainedValue()
    let functionName = String(cString: name)
    
    guard let function = library.makeFunction(name: functionName) else {
        return nil
    }
    return Unmanaged.passRetained(function).toOpaque()
}

// MARK: - Compute Pipeline Functions

@_cdecl("metal_device_make_compute_pipeline_state")
public func metal_device_make_compute_pipeline_state(_ devicePtr: UnsafeMutableRawPointer,
                                                   _ functionPtr: UnsafeMutableRawPointer) -> UnsafeMutableRawPointer? {
    let device = Unmanaged<MTLDevice>.fromOpaque(devicePtr).takeUnretainedValue()
    let function = Unmanaged<MTLFunction>.fromOpaque(functionPtr).takeUnretainedValue()
    
    do {
        let pipelineState = try device.makeComputePipelineState(function: function)
        return Unmanaged.passRetained(pipelineState).toOpaque()
    } catch {
        print("Error creating compute pipeline state: \(error)")
        return nil
    }
}

// MARK: - Command Encoder Functions

@_cdecl("metal_command_buffer_make_compute_command_encoder")
public func metal_command_buffer_make_compute_command_encoder(_ commandBufferPtr: UnsafeMutableRawPointer) -> UnsafeMutableRawPointer? {
    let commandBuffer = Unmanaged<MTLCommandBuffer>.fromOpaque(commandBufferPtr).takeUnretainedValue()
    guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
        return nil
    }
    return Unmanaged.passRetained(encoder).toOpaque()
}

@_cdecl("metal_compute_command_encoder_set_compute_pipeline_state")
public func metal_compute_command_encoder_set_compute_pipeline_state(_ encoderPtr: UnsafeMutableRawPointer,
                                                                   _ pipelinePtr: UnsafeMutableRawPointer) {
    let encoder = Unmanaged<MTLComputeCommandEncoder>.fromOpaque(encoderPtr).takeUnretainedValue()
    let pipelineState = Unmanaged<MTLComputePipelineState>.fromOpaque(pipelinePtr).takeUnretainedValue()
    encoder.setComputePipelineState(pipelineState)
}

@_cdecl("metal_compute_command_encoder_set_buffer")
public func metal_compute_command_encoder_set_buffer(_ encoderPtr: UnsafeMutableRawPointer,
                                                   _ bufferPtr: UnsafeMutableRawPointer,
                                                   _ offset: UInt64,
                                                   _ index: Int32) {
    let encoder = Unmanaged<MTLComputeCommandEncoder>.fromOpaque(encoderPtr).takeUnretainedValue()
    let buffer = Unmanaged<MTLBuffer>.fromOpaque(bufferPtr).takeUnretainedValue()
    encoder.setBuffer(buffer, offset: Int(offset), index: Int(index))
}

// Add missing shim: dispatchThreadgroups(threadgroupsPerGrid:threadsPerThreadgroup:)
@_cdecl("metal_compute_command_encoder_dispatch_threadgroups")
public func metal_compute_command_encoder_dispatch_threadgroups(_ encoderPtr: UnsafeMutableRawPointer,
                                                              _ groupsX: Int32, _ groupsY: Int32, _ groupsZ: Int32,
                                                              _ threadsPerGroupX: Int32, _ threadsPerGroupY: Int32, _ threadsPerGroupZ: Int32) {
    let encoder = Unmanaged<MTLComputeCommandEncoder>.fromOpaque(encoderPtr).takeUnretainedValue()
    let threadgroupsPerGrid = MTLSize(width: Int(groupsX), height: Int(groupsY), depth: Int(groupsZ))
    let threadsPerThreadgroup = MTLSize(width: Int(threadsPerGroupX), height: Int(threadsPerGroupY), depth: Int(threadsPerGroupZ))
    encoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
}

// Add missing: setThreadgroupMemoryLength(length:index:)
@_cdecl("metal_compute_command_encoder_set_threadgroup_memory_length")
public func metal_compute_command_encoder_set_threadgroup_memory_length(_ encoderPtr: UnsafeMutableRawPointer,
                                                                      _ length: UInt64,
                                                                      _ index: Int32) {
    let encoder = Unmanaged<MTLComputeCommandEncoder>.fromOpaque(encoderPtr).takeUnretainedValue()
    encoder.setThreadgroupMemoryLength(Int(length), index: Int(index))
}

@_cdecl("metal_compute_command_encoder_dispatch_threads")
public func metal_compute_command_encoder_dispatch_threads(_ encoderPtr: UnsafeMutableRawPointer,
                                                         _ threadsX: Int32, _ threadsY: Int32, _ threadsZ: Int32,
                                                         _ threadgroupX: Int32, _ threadgroupY: Int32, _ threadgroupZ: Int32) {
    let encoder = Unmanaged<MTLComputeCommandEncoder>.fromOpaque(encoderPtr).takeUnretainedValue()
    let threadsPerGrid = MTLSize(width: Int(threadsX), height: Int(threadsY), depth: Int(threadsZ))
    let threadsPerThreadgroup = MTLSize(width: Int(threadgroupX), height: Int(threadgroupY), depth: Int(threadgroupZ))
    
    encoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
}

@_cdecl("metal_compute_command_encoder_end_encoding")
public func metal_compute_command_encoder_end_encoding(_ encoderPtr: UnsafeMutableRawPointer) {
    let encoder = Unmanaged<MTLComputeCommandEncoder>.fromOpaque(encoderPtr).takeUnretainedValue()
    encoder.endEncoding()
}

// MARK: - Command Buffer Execution

@_cdecl("metal_command_buffer_commit")
public func metal_command_buffer_commit(_ commandBufferPtr: UnsafeMutableRawPointer) {
    let commandBuffer = Unmanaged<MTLCommandBuffer>.fromOpaque(commandBufferPtr).takeUnretainedValue()
    commandBuffer.commit()
}

@_cdecl("metal_command_buffer_wait_until_completed")
public func metal_command_buffer_wait_until_completed(_ commandBufferPtr: UnsafeMutableRawPointer) {
    let commandBuffer = Unmanaged<MTLCommandBuffer>.fromOpaque(commandBufferPtr).takeUnretainedValue()
    commandBuffer.waitUntilCompleted()
}

// MARK: - Memory Management Helper Functions

@_cdecl("metal_release_object")
public func metal_release_object(_ objectPtr: UnsafeMutableRawPointer) {
    // Generic release for any Metal object
    _ = Unmanaged<AnyObject>.fromOpaque(objectPtr).takeRetainedValue()
}

// MARK: - Specific Resource Release Functions

@_cdecl("metal_buffer_release")
public func metal_buffer_release(_ bufferPtr: UnsafeMutableRawPointer) {
    _ = Unmanaged<MTLBuffer>.fromOpaque(bufferPtr).takeRetainedValue()
}

@_cdecl("metal_library_release")
public func metal_library_release(_ libraryPtr: UnsafeMutableRawPointer) {
    _ = Unmanaged<MTLLibrary>.fromOpaque(libraryPtr).takeRetainedValue()
}

@_cdecl("metal_function_release")
public func metal_function_release(_ functionPtr: UnsafeMutableRawPointer) {
    _ = Unmanaged<MTLFunction>.fromOpaque(functionPtr).takeRetainedValue()
}

@_cdecl("metal_compute_pipeline_state_release")
public func metal_compute_pipeline_state_release(_ pipelinePtr: UnsafeMutableRawPointer) {
    _ = Unmanaged<MTLComputePipelineState>.fromOpaque(pipelinePtr).takeRetainedValue()
}

@_cdecl("metal_command_queue_release")
public func metal_command_queue_release(_ queuePtr: UnsafeMutableRawPointer) {
    _ = Unmanaged<MTLCommandQueue>.fromOpaque(queuePtr).takeRetainedValue()
}

@_cdecl("metal_command_buffer_release")
public func metal_command_buffer_release(_ bufferPtr: UnsafeMutableRawPointer) {
    _ = Unmanaged<MTLCommandBuffer>.fromOpaque(bufferPtr).takeRetainedValue()
}

@_cdecl("metal_compute_command_encoder_release")
public func metal_compute_command_encoder_release(_ encoderPtr: UnsafeMutableRawPointer) {
    _ = Unmanaged<MTLComputeCommandEncoder>.fromOpaque(encoderPtr).takeRetainedValue()
}

// MARK: - Error Handling

@_cdecl("metal_get_last_error")
public func metal_get_last_error() -> UnsafePointer<CChar>? {
    // In a production implementation, you'd maintain an error state
    return nil
}
