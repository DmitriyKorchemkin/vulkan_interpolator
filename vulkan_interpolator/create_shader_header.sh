#!/bin/bash -xe

USAGE="Usage: $0 --output output_header_name --namespace target_namespace shader_1.spv ... shader_N.spv"

while [[ $# -gt 0 ]]
do
  key="$1"
  case $key in
    --output)
      output="$2"
      shift
      shift
      ;;
    --namespace)
      namespace="$2"
      shift
      shift
      ;;
    *)
      shaders+=("$1")
      shift
      ;;
  esac
done

if [ -z "$output" ]; then
  echo "--output is required"
  echo "$USAGE"
  exit -1;
fi

mkdir -p $(dirname "$output")
guard=$(basename "$output" | sed 's/[^[:alnum:]]/_/g')
echo "#ifndef $guard
#define $guard
#include <string_view>" > "$output"

if [ ! -z "$namespace" ]; then
  echo "namespace $namespace {" >> "$output"
fi
echo "using namespace std::literals::string_view_literals;" >> "$output"

for shader in "${shaders[@]}"; do
  echo "constexpr std::string_view $(basename "$shader" | sed 's/[^[:alnum:]]/_/g') = \"$(cat $shader | xxd -p | tr -d $'\n' | sed 's/\([0-9a-f]\{2\}\)/\\x\1/g')\"sv;" >> "$output"
done

if [ ! -z "$namespace" ]; then
  echo "} // namespace $namespace " >> "$output"
fi

echo "#endif // $guard" >> "$output"
