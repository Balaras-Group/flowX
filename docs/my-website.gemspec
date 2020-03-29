# encoding: utf-8

Gem::Specification.new do |s|
  s.name          = "my-website"
  s.version       = "0.0.0"
  s.license       = "CC0-1.0"
  s.authors       = ["Akash Dhruv"]
  s.summary       = "This a theme for my website"

  s.files         = `git ls-files -z`.split("\x0").select do |f|
    f.match(%r{^((_includes|_layouts|_sass|assets)/|(LICENSE|README)((\.(txt|md|markdown)|$)))}i)
  end

  s.platform      = Gem::Platform::RUBY
  s.add_runtime_dependency "jekyll", "~> 3.3"
end
