# System Architecture

## Configuration Flow
The system utilizes a "validate-on-laod" strategy to ensure type safety.
1. **Load**: ConfigLoader reads raw YAML files in the specified subdirectory.
    * Decision: Malformed YAML files are logged and skipped.
2. **Validate**: Pydantic models (schemas/) check data integrity immediately.
    * Decision: Invalid files are logged and skipped, preventing full crash but ensuring bad data doesn't enter the system.
3. **Execute**: ExecutionService receives only valid configuration objects. 
    * Decision: If the requested agent is not contained in the valid config, execution stops.

| Step | Component | Responsibility | Decision Logic |
| :--- | :--- | :--- | :--- |
| **Load** | `ConfigLoader` | Reads raw YAML files in the specified subdirectory. | **Skip & Log** Malformed YAML files are skipped to prevent a full crash. |
| **Validate** | `Pydantic Models` | Checks data integrity against schemas immediately. | **Skip & Log:** Files containing invalid schemas are skipped to prevent a full crash. |
| **Execute** | `ExecutionService` | Receives and processes only valid configuration objects. | **Raise Error:** If the requested agent is not in the valid config, execution stops. |